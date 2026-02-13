package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/internal/tokenizer"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type diffStats struct {
	MaxAbs       float64
	MeanAbs      float64
	RMSE         float64
	Cosine       float64
	Top1A        int
	Top1B        int
	Top1Match    bool
	Top1Delta    float64
	Top5Overlap  int
	VectorLength int
}

func main() {
	var (
		bf16Path       string
		k4Path         string
		prompt         string
		system         string
		steps          int
		maxContext     int
		noTemplate     bool
		showTokens     bool
		comparePrefill bool
		topK           int
		showTop1Text   bool

		tokenizerJSON   string
		tokenizerConfig string
		chatTemplate    string
	)

	flag.StringVar(&bf16Path, "bf16", "", "path to BF16 .mcf")
	flag.StringVar(&k4Path, "k4", "", "path to K4 .mcf")
	flag.StringVar(&prompt, "prompt", "", "prompt text")
	flag.StringVar(&system, "system", "", "optional system prompt")
	flag.IntVar(&steps, "steps", 8, "number of generated steps to compare")
	flag.IntVar(&maxContext, "max-context", 4096, "max context length")
	flag.BoolVar(&noTemplate, "no-template", false, "disable chat template rendering")
	flag.BoolVar(&showTokens, "show-tokens", false, "print input tokens")
	flag.BoolVar(&comparePrefill, "compare-prefill", false, "include prefill tokens in diff output")
	flag.IntVar(&topK, "topk", 5, "top-k overlap to report")
	flag.BoolVar(&showTop1Text, "show-top1-text", false, "decode top1 tokens for each step")
	flag.StringVar(&tokenizerJSON, "tokenizer-json", "", "override path to tokenizer.json")
	flag.StringVar(&tokenizerConfig, "tokenizer-config", "", "override path to tokenizer_config.json")
	flag.StringVar(&chatTemplate, "chat-template", "", "override chat_template")
	flag.Parse()

	if bf16Path == "" || k4Path == "" {
		fmt.Fprintln(os.Stderr, "--bf16 and --k4 are required")
		os.Exit(2)
	}
	if steps < 0 {
		fmt.Fprintln(os.Stderr, "--steps must be >= 0")
		os.Exit(2)
	}

	bf16File, err := mcfstore.Open(bf16Path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open bf16:", err)
		os.Exit(1)
	}
	defer func() { _ = bf16File.Close() }()

	k4File, err := mcfstore.Open(k4Path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open k4:", err)
		os.Exit(1)
	}
	defer func() { _ = k4File.Close() }()

	bfCfgBytes := bf16File.SectionData(mcf.SectionHFConfigJSON)
	if len(bfCfgBytes) == 0 {
		fmt.Fprintln(os.Stderr, "bf16: missing config.json")
		os.Exit(1)
	}
	k4CfgBytes := k4File.SectionData(mcf.SectionHFConfigJSON)
	if len(k4CfgBytes) == 0 {
		fmt.Fprintln(os.Stderr, "k4: missing config.json")
		os.Exit(1)
	}

	bfModel, err := simd.LoadModelMCF(bf16File, bfCfgBytes, maxContext, simd.LoadModelOptions{})
	if err != nil {
		fmt.Fprintln(os.Stderr, "bf16 load:", err)
		os.Exit(1)
	}
	k4Model, err := simd.LoadModelMCF(k4File, k4CfgBytes, maxContext, simd.LoadModelOptions{})
	if err != nil {
		fmt.Fprintln(os.Stderr, "k4 load:", err)
		os.Exit(1)
	}

	tok, tokCfg, cfgBytes, arch := loadTokenizer(bf16File, tokenizerJSON, tokenizerConfig, bfCfgBytes, bfModel.Config.Arch)
	if tok == nil {
		fmt.Fprintln(os.Stderr, "tokenizer: failed to load")
		os.Exit(1)
	}

	effectiveTemplate, _ := resolveChatTemplate(chatTemplate, tokCfg, arch, cfgBytes)
	rendered := renderPrompt(prompt, system, tok, tokCfg, effectiveTemplate, noTemplate, arch)
	if rendered == "" {
		fmt.Fprintln(os.Stderr, "prompt is empty after rendering")
		os.Exit(2)
	}

	ids, err := tok.Encode(rendered)
	if err != nil {
		fmt.Fprintln(os.Stderr, "encode:", err)
		os.Exit(1)
	}
	if showTokens {
		fmt.Fprintf(os.Stderr, "Input tokens (%d): %s\n", len(ids), joinInts(ids))
	}

	fmt.Printf("BF16=%s\n", bf16Path)
	fmt.Printf("K4=%s\n", k4Path)
	fmt.Printf("Prompt tokens=%d steps=%d compare_prefill=%v\n", len(ids), steps, comparePrefill)

	var (
		bfLogits []float32
		k4Logits []float32
	)

	acc := diffAccumulator{}

	for i, id := range ids {
		bfLogits, err = bfModel.ForwardToken(id)
		if err != nil {
			fmt.Fprintf(os.Stderr, "bf16 prefill %d: %v\n", i, err)
			os.Exit(1)
		}
		k4Logits, err = k4Model.ForwardToken(id)
		if err != nil {
			fmt.Fprintf(os.Stderr, "k4 prefill %d: %v\n", i, err)
			os.Exit(1)
		}
		if comparePrefill {
			stats := diffLogits(bfLogits, k4Logits, topK)
			acc.add(stats)
			printStep("prefill", i, id, stats, tok, showTop1Text)
		}
	}

	for step := 0; step < steps; step++ {
		if len(bfLogits) == 0 || len(k4Logits) == 0 {
			fmt.Fprintln(os.Stderr, "empty logits")
			os.Exit(1)
		}
		stats := diffLogits(bfLogits, k4Logits, topK)
		acc.add(stats)
		next := stats.Top1A
		printStep("gen", step, next, stats, tok, showTop1Text)

		bfLogits, err = bfModel.ForwardToken(next)
		if err != nil {
			fmt.Fprintf(os.Stderr, "bf16 gen %d: %v\n", step, err)
			os.Exit(1)
		}
		k4Logits, err = k4Model.ForwardToken(next)
		if err != nil {
			fmt.Fprintf(os.Stderr, "k4 gen %d: %v\n", step, err)
			os.Exit(1)
		}
	}

	if acc.count > 0 {
		fmt.Println()
		fmt.Printf("Summary steps=%d max_abs=%.6g mean_abs=%.6g rmse=%.6g cos=%.6g top1_match=%.2f%% top5_overlap=%.2f\n",
			acc.count,
			acc.maxAbs,
			acc.meanAbs/float64(acc.count),
			acc.rmse/float64(acc.count),
			acc.cos/float64(acc.count),
			100.0*float64(acc.top1Match)/float64(acc.count),
			float64(acc.top5Overlap)/float64(acc.count),
		)
	}
}

type diffAccumulator struct {
	count       int
	maxAbs      float64
	meanAbs     float64
	rmse        float64
	cos         float64
	top1Match   int
	top5Overlap int
}

func (a *diffAccumulator) add(s diffStats) {
	a.count++
	if s.MaxAbs > a.maxAbs {
		a.maxAbs = s.MaxAbs
	}
	a.meanAbs += s.MeanAbs
	a.rmse += s.RMSE
	a.cos += s.Cosine
	if s.Top1Match {
		a.top1Match++
	}
	a.top5Overlap += s.Top5Overlap
}

func diffLogits(a, b []float32, topK int) diffStats {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return diffStats{}
	}
	var (
		sumAbs float64
		sumSq  float64
		dot    float64
		normA  float64
		normB  float64
		maxAbs float64
	)
	top1A := 0
	top1B := 0
	maxA := a[0]
	maxB := b[0]
	for i := 0; i < n; i++ {
		da := float64(a[i])
		db := float64(b[i])
		diff := da - db
		if diff < 0 {
			diff = -diff
		}
		sumAbs += diff
		sumSq += diff * diff
		if diff > maxAbs {
			maxAbs = diff
		}
		dot += da * db
		normA += da * da
		normB += db * db
		if a[i] > maxA {
			maxA = a[i]
			top1A = i
		}
		if b[i] > maxB {
			maxB = b[i]
			top1B = i
		}
	}
	cos := 0.0
	if normA > 0 && normB > 0 {
		cos = dot / (math.Sqrt(normA) * math.Sqrt(normB))
	}

	top5Overlap := 0
	if topK > 1 {
		ta := topKIndices(a[:n], topK)
		tb := topKIndices(b[:n], topK)
		seen := make(map[int]struct{}, len(ta))
		for _, idx := range ta {
			seen[idx] = struct{}{}
		}
		for _, idx := range tb {
			if _, ok := seen[idx]; ok {
				top5Overlap++
			}
		}
	}

	return diffStats{
		MaxAbs:       maxAbs,
		MeanAbs:      sumAbs / float64(n),
		RMSE:         math.Sqrt(sumSq / float64(n)),
		Cosine:       cos,
		Top1A:        top1A,
		Top1B:        top1B,
		Top1Match:    top1A == top1B,
		Top1Delta:    float64(a[top1A] - b[top1B]),
		Top5Overlap:  top5Overlap,
		VectorLength: n,
	}
}

func topKIndices(vals []float32, k int) []int {
	if k <= 0 {
		return nil
	}
	if k > len(vals) {
		k = len(vals)
	}
	idx := make([]int, 0, k)
	score := make([]float32, 0, k)
	for i, v := range vals {
		pos := len(score)
		for j, s := range score {
			if v > s {
				pos = j
				break
			}
		}
		if pos == len(score) {
			if len(score) < k {
				score = append(score, v)
				idx = append(idx, i)
			}
			continue
		}
		score = append(score, 0)
		idx = append(idx, 0)
		copy(score[pos+1:], score[pos:])
		copy(idx[pos+1:], idx[pos:])
		score[pos] = v
		idx[pos] = i
		if len(score) > k {
			score = score[:k]
			idx = idx[:k]
		}
	}
	return idx
}

func printStep(phase string, step int, token int, stats diffStats, tok tokenizer.Tokenizer, showTop1Text bool) {
	top1Text := ""
	if showTop1Text && tok != nil {
		if s, err := tok.Decode([]int{stats.Top1A}); err == nil {
			top1Text = strings.ReplaceAll(s, "\n", "\\n")
		}
	}
	fmt.Printf(
		"%s[%d] tok=%d top1_bf16=%d top1_k4=%d match=%v top5=%d max_abs=%.6g mean_abs=%.6g rmse=%.6g cos=%.6g%s\n",
		phase,
		step,
		token,
		stats.Top1A,
		stats.Top1B,
		stats.Top1Match,
		stats.Top5Overlap,
		stats.MaxAbs,
		stats.MeanAbs,
		stats.RMSE,
		stats.Cosine,
		formatTop1Text(top1Text),
	)
}

func formatTop1Text(s string) string {
	if s == "" {
		return ""
	}
	return fmt.Sprintf(" top1_text=%q", s)
}

func renderPrompt(prompt, system string, tok tokenizer.Tokenizer, cfg tokenizer.TokenizerConfig, template string, noTemplate bool, arch string) string {
	msgs := make([]tokenizer.Message, 0, 2)
	if system != "" {
		msgs = append(msgs, tokenizer.Message{Role: "system", Content: system})
	}
	if prompt != "" {
		msgs = append(msgs, tokenizer.Message{Role: "user", Content: prompt})
	}

	if !noTemplate {
		bosToken := ""
		if t, ok := tok.(interface{ TokenString(int) string }); ok {
			bosToken = t.TokenString(cfg.BOSTokenID)
		}
		if s, ok, err := tokenizer.RenderPromptTemplate(template, bosToken, cfg.TokenString(cfg.EOSTokenID), cfg.AddBOS, msgs, nil, true, arch); err == nil && ok {
			return s
		}
	}

	if len(msgs) > 0 {
		if text, ok := tokenizer.MessageText(msgs[len(msgs)-1]); ok {
			return text
		}
	}
	return ""
}

func loadTokenizer(mf *mcfstore.File, tokJSONPath, tokCfgPath string, cfgBytes []byte, arch string) (tokenizer.Tokenizer, tokenizer.TokenizerConfig, []byte, string) {
	var tokCfg tokenizer.TokenizerConfig
	if tokJSONPath != "" && fileExists(tokJSONPath) {
		tokJSON, err := os.ReadFile(tokJSONPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "read tokenizer.json:", err)
			return nil, tokCfg, cfgBytes, arch
		}
		var tokCfgBytes []byte
		if tokCfgPath != "" && fileExists(tokCfgPath) {
			tokCfgBytes, err = os.ReadFile(tokCfgPath)
			if err != nil {
				fmt.Fprintln(os.Stderr, "read tokenizer_config.json:", err)
				return nil, tokCfg, cfgBytes, arch
			}
		}
		hfTok, err := tokenizer.LoadHFTokenizerBytes(tokJSON, tokCfgBytes)
		if err != nil {
			fmt.Fprintln(os.Stderr, "load tokenizer.json:", err)
			return nil, tokCfg, cfgBytes, arch
		}
		if parsed, err := tokenizer.ParseHFTokenizerConfigBytes(tokJSON, tokCfgBytes); err == nil {
			tokCfg = parsed
		}
		tokCfg.BOSTokenID = hfTok.BOSID()
		tokCfg.AddBOS = hfTok.AddBOS()
		tokCfg.EOSTokenID = hfTok.EOSID()
		return hfTok, tokCfg, cfgBytes, arch
	}

	tokJSON := mf.SectionData(mcf.SectionTokenizerJSON)
	if len(tokJSON) == 0 {
		fmt.Fprintln(os.Stderr, "tokenizer.json not found in MCF")
		return nil, tokCfg, cfgBytes, arch
	}
	var tokCfgBytes []byte
	if tokCfgPath != "" && fileExists(tokCfgPath) {
		raw, err := os.ReadFile(tokCfgPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "read tokenizer_config.json:", err)
			return nil, tokCfg, cfgBytes, arch
		}
		tokCfgBytes = raw
	} else {
		tokCfgBytes = mf.SectionData(mcf.SectionTokenizerConfigJSON)
	}

	hfTok, err := tokenizer.LoadHFTokenizerBytes(tokJSON, tokCfgBytes)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load tokenizer.json:", err)
		return nil, tokCfg, cfgBytes, arch
	}
	if parsed, err := tokenizer.ParseHFTokenizerConfigBytes(tokJSON, tokCfgBytes); err == nil {
		tokCfg = parsed
	}
	tokCfg.BOSTokenID = hfTok.BOSID()
	tokCfg.AddBOS = hfTok.AddBOS()
	tokCfg.EOSTokenID = hfTok.EOSID()
	return hfTok, tokCfg, cfgBytes, arch
}

func resolveChatTemplate(override string, cfg tokenizer.TokenizerConfig, arch string, hfConfig []byte) (string, string) {
	template := strings.TrimSpace(override)
	source := ""
	switch {
	case template != "":
		source = "flag"
	case strings.TrimSpace(cfg.ChatTemplate) != "":
		template = cfg.ChatTemplate
		source = "tokenizer_config"
	default:
		if inferred, ok := model.InferChatTemplate(arch, hfConfig); ok {
			template = inferred
			source = "model-default"
		} else {
			return "", "none"
		}
	}

	if len(template) < 256 && fileExists(template) {
		if raw, err := os.ReadFile(template); err == nil && len(raw) > 0 {
			template = string(raw)
			source += ":file"
		}
	}
	return template, source
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

func joinInts(ids []int) string {
	if len(ids) == 0 {
		return "[]"
	}
	var b strings.Builder
	b.WriteByte('[')
	for i, id := range ids {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(fmt.Sprintf("%d", id))
	}
	b.WriteByte(']')
	return b.String()
}
