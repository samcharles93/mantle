package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

// Swarm handles the multi-worker coordination and execution.
type Swarm struct {
	Manager    *StateManager
	Engine     inference.Engine
	Workers    int
	MaxSteps   int
	Interactive bool
	Verifier   string
}

func (s *Swarm) Run(ctx context.Context, goal string) error {
	log := logger.FromContext(ctx)

	// Step 1: Initialize Plan
	plan, err := s.generatePlan(ctx, goal)
	if err != nil {
		return fmt.Errorf("generate plan: %w", err)
	}
	if err := s.Manager.SavePlan(plan); err != nil {
		return fmt.Errorf("save plan: %w", err)
	}

	log.Info("Plan generated", "goal", goal, "tasks", len(plan.TaskIDs))
	for _, id := range plan.TaskIDs {
		log.Info("Task added", "id", id)
	}

	// Step 2: Spawn Workers
	errChan := make(chan error, s.Workers)
	for i := 0; i < s.Workers; i++ {
		workerID := fmt.Sprintf("worker-%d", i)
		go func() {
			errChan <- s.workerLoop(ctx, workerID)
		}()
	}

	// Step 3: Monitor until completion
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-errChan:
			if err != nil {
				return fmt.Errorf("worker died: %w", err)
			}
		case <-ticker.C:
			allDone, err := s.checkCompletion()
			if err != nil {
				log.Error("check completion failed", "error", err)
				continue
			}
			if allDone {
				log.Info("Swarm mission completed successfully!")
				return nil
			}
		}
	}
}

func (s *Swarm) generatePlan(ctx context.Context, goal string) (*Plan, error) {
	prompt := fmt.Sprintf("Break down the following goal into a sequence of small, manageable tasks for independent workers.\nGoal: %s\n\nOutput JSON format ONLY:\n{\n  \"tasks\": [\n    {\"id\": \"task_id\", \"description\": \"task description\", \"dependencies\": [\"dependency_id\"]}\n  ]\n}\n", goal)

	req := inference.Request{
		Messages: []tokenizer.Message{
			{Role: "system", Content: "You are a master architect. Break goals into parallelizable tasks."},
			{Role: "user", Content: prompt},
		},
	}

	res, err := s.Engine.Generate(ctx, &req, nil)
	if err != nil {
		return nil, err
	}

	jsonStart := strings.Index(res.Text, "{")
	jsonEnd := strings.LastIndex(res.Text, "}")
	if jsonStart == -1 || jsonEnd == -1 {
		return nil, fmt.Errorf("failed to parse plan JSON from model output: %s", res.Text)
	}
	jsonStr := res.Text[jsonStart : jsonEnd+1]

	var wrapper struct {
		Tasks []struct {
			ID           string   `json:"id"`
			Description  string   `json:"description"`
			Dependencies []string `json:"dependencies"`
		} `json:"tasks"`
	}
	if err := json.Unmarshal([]byte(jsonStr), &wrapper); err != nil {
		return nil, fmt.Errorf("unmarshal plan JSON: %w", err)
	}

	plan := &Plan{
		Goal:      goal,
		CreatedAt: time.Now(),
		Status:    "in_progress",
	}

	for _, t := range wrapper.Tasks {
		plan.TaskIDs = append(plan.TaskIDs, t.ID)
		task := &Task{
			ID:           t.ID,
			Description:  t.Description,
			Dependencies: t.Dependencies,
			Status:       TaskPending,
		}
		if err := s.Manager.SaveTask(task); err != nil {
			return nil, err
		}
	}

	return plan, nil
}

func (s *Swarm) workerLoop(ctx context.Context, workerID string) error {
	log := logger.FromContext(ctx)
	log.Info("Worker started", "worker_id", workerID)

	registry := NewRegistry()
	workspace, _ := os.Getwd()
	registry.Register(&WriteFileTool{Workspace: workspace})
	registry.Register(&ReadFileTool{Workspace: workspace})
	registry.Register(&ExecuteShellTool{Workspace: workspace})
	registry.Register(&ListDirectoryTool{Workspace: workspace})

	for {
		select {
		case <-ctx.Done():
			return nil
		default:
			taskID, err := s.findAvailableTask()
			if err != nil {
				log.Error("find task failed", "worker_id", workerID, "error", err)
				time.Sleep(1 * time.Second)
				continue
			}

			if taskID == "" {
				time.Sleep(1 * time.Second)
				continue
			}

			locked, err := s.Manager.TryLockTask(taskID, workerID)
			if err != nil || !locked {
				continue
			}

			log.Info("Task locked", "worker_id", workerID, "task_id", taskID)
			if err := s.executeTask(ctx, workerID, taskID, registry); err != nil {
				log.Error("task execution failed", "worker_id", workerID, "task_id", taskID, "error", err)
			}

			_ = s.Manager.UnlockTask(taskID)
		}
	}
}

func (s *Swarm) findAvailableTask() (string, error) {
	tasks, err := s.Manager.ListTasks()
	if err != nil {
		return "", err
	}

	completed := make(map[string]bool)
	for _, t := range tasks {
		if t.Status == TaskCompleted {
			completed[t.ID] = true
		}
	}

	for _, t := range tasks {
		if t.Status != TaskPending && t.Status != TaskFailed {
			continue
		}

		if t.Attempts >= 3 {
			continue
		}

		ready := true
		for _, dep := range t.Dependencies {
			if !completed[dep] {
				ready = false
				break
			}
		}

		if ready {
			return t.ID, nil
		}
	}

	return "", nil
}

func (s *Swarm) executeTask(ctx context.Context, workerID string, taskID string, registry *Registry) error {
	log := logger.FromContext(ctx)
	task, err := s.Manager.LoadTask(taskID)
	if err != nil {
		return err
	}

	task.Status = TaskInProgress
	task.LastWorkerID = workerID
	task.Attempts++
	_ = s.Manager.SaveTask(task)

	l := &Loop{
		Engine:      s.Engine,
		Registry:    registry,
		MaxSteps:    s.MaxSteps,
		Interactive: s.Interactive,
		// Out is nil: swarm workers are quiet; the swarm coordinator logs via structured logger.
	}

	l.Messages = append(l.Messages, tokenizer.Message{
		Role: "system",
		Content: fmt.Sprintf("Goal: %s\nTask: %s\nWorker: %s", task.Description, taskID, workerID),
	})

	res, err := l.Run(ctx, task.Description)
	if err != nil || !res.Success {
		task.Status = TaskFailed
		_ = s.Manager.SaveTask(task)
		return err
	}

	task.Status = TaskCompleted
	task.ContextSummary = res.Output
	_ = s.Manager.SaveTask(task)
	log.Info("Task completed", "worker_id", workerID, "task_id", taskID)
	return nil
}

func (s *Swarm) checkCompletion() (bool, error) {
	tasks, err := s.Manager.ListTasks()
	if err != nil {
		return false, err
	}
	if len(tasks) == 0 {
		return false, nil
	}
	for _, t := range tasks {
		if t.Status != TaskCompleted {
			return false, nil
		}
	}
	return true, nil
}
