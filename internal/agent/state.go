package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// TaskStatus defines the possible states for a subtask.
type TaskStatus string

const (
	TaskPending    TaskStatus = "pending"
	TaskInProgress TaskStatus = "in_progress"
	TaskCompleted  TaskStatus = "completed"
	TaskFailed     TaskStatus = "failed"
)

// Plan represents the master registry for a swarm mission.
type Plan struct {
	Goal      string    `json:"goal"`
	CreatedAt time.Time `json:"created_at"`
	Status    string    `json:"status"` // pending, in_progress, completed, failed
	TaskIDs   []string  `json:"task_ids"`
}

// Task represents an individual unit of work within a plan.
type Task struct {
	ID             string     `json:"id"`
	Description    string     `json:"description"`
	Dependencies   []string   `json:"dependencies"`
	Status         TaskStatus `json:"status"`
	Attempts       int        `json:"attempts"`
	LastWorkerID   string     `json:"last_worker_id,omitempty"`
	ContextSummary string     `json:"context_summary,omitempty"`
}

// StateManager handles persistence of the swarm state using a directory of JSON files.
type StateManager struct {
	Dir string
}

func NewStateManager(dir string) (*StateManager, error) {
	if err := os.MkdirAll(filepath.Join(dir, "tasks"), 0755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Join(dir, "history"), 0755); err != nil {
		return nil, err
	}
	return &StateManager{Dir: dir}, nil
}

func (s *StateManager) SavePlan(p *Plan) error {
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(s.Dir, "plan.json"), data, 0644)
}

func (s *StateManager) LoadPlan() (*Plan, error) {
	data, err := os.ReadFile(filepath.Join(s.Dir, "plan.json"))
	if err != nil {
		return nil, err
	}
	var p Plan
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, err
	}
	return &p, nil
}

func (s *StateManager) SaveTask(t *Task) error {
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(s.Dir, "tasks", t.ID+".json"), data, 0644)
}

func (s *StateManager) LoadTask(id string) (*Task, error) {
	data, err := os.ReadFile(filepath.Join(s.Dir, "tasks", id+".json"))
	if err != nil {
		return nil, err
	}
	var t Task
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, err
	}
	return &t, nil
}

func (s *StateManager) ListTasks() ([]*Task, error) {
	files, err := os.ReadDir(filepath.Join(s.Dir, "tasks"))
	if err != nil {
		return nil, err
	}
	var tasks []*Task
	for _, f := range files {
		if filepath.Ext(f.Name()) == ".json" {
			id := f.Name()[:len(f.Name())-len(".json")]
			t, err := s.LoadTask(id)
			if err != nil {
				continue
			}
			tasks = append(tasks, t)
		}
	}
	return tasks, nil
}

func (s *StateManager) TryLockTask(id string, workerID string) (bool, error) {
	lockPath := filepath.Join(s.Dir, "tasks", id+".lock")
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		if os.IsExist(err) {
			return false, nil
		}
		return false, err
	}
	defer f.Close()
	fmt.Fprintf(f, "%s\n%d", workerID, time.Now().Unix())
	return true, nil
}

func (s *StateManager) UnlockTask(id string) error {
	return os.Remove(filepath.Join(s.Dir, "tasks", id+".lock"))
}

func (s *StateManager) LogAction(workerID string, taskID string, action string, observation string) error {
	logEntry := map[string]any{
		"timestamp": time.Now().Format(time.RFC3339),
		"worker_id": workerID,
		"task_id":   taskID,
		"action":    action,
		"observation": observation,
	}
	data, _ := json.Marshal(logEntry)
	f, err := os.OpenFile(filepath.Join(s.Dir, "history", "worker_actions.jsonl"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(data, '\n'))
	return err
}
