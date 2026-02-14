package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/urfave/cli/v3"

	"github.com/samcharles93/mantle/internal/tokenizer"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type tokenizerJSON struct {
	Model struct {
		Type   string         `json:"type"`
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

type tokenizerConfigJSON struct {
	ChatTemplate string `json:"chat_template"`
	AddBOS       bool   `json:"add_bos_token"`
	AddEOS       bool   `json:"add_eos_token"`
	BOS          string `json:"bos_token"`
	EOS          string `json:"eos_token"`
	UNK          string `json:"unk_token"`
	PAD          string `json:"pad_token"`
	BOSID        *int   `json:"bos_token_id"`
	EOSID        *int   `json:"eos_token_id"`
	UNKID        *int   `json:"unk_token_id"`
	PADID        *int   `json:"pad_token_id"`
}

func inspectCmd() *cli.Command {
	var (
		modelPath        string
		showAll          bool
		showSections     bool
		showTensors      bool
		showQuant        bool
		showTokenizer    bool
		showChat         bool
		showVocab        bool
		showTokenizerJS  bool
		showTokenizerCfg bool
		showHFConfig     bool
		showGenConfig    bool
		showVocabJSON    bool
		showMerges       bool
		showModelInfo    bool
		showTensorData   bool
		tensorLimit      int
		vocabLimit       int
		tensorFilter     string
	)

	return &cli.Command{
		Name:  "inspect",
		Usage: "Inspect the contents of an .mcf model container",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:        "model",
				Aliases:     []string{"m"},
				Usage:       "path to .mcf file",
				Destination: &modelPath,
				Required:    true,
			},
			&cli.BoolFlag{Name: "all", Usage: "show all sections and raw resources", Destination: &showAll},
			&cli.BoolFlag{Name: "sections", Usage: "show section directory", Destination: &showSections},
			&cli.BoolFlag{Name: "tensors", Usage: "list tensor index", Destination: &showTensors},
			&cli.BoolFlag{Name: "quant", Usage: "show quantization info", Destination: &showQuant},
			&cli.BoolFlag{Name: "tokenizer", Usage: "show tokenizer summary", Destination: &showTokenizer},
			&cli.BoolFlag{Name: "chat-template", Usage: "print embedded chat template", Destination: &showChat},
			&cli.BoolFlag{Name: "vocab", Usage: "list vocab entries", Destination: &showVocab},
			&cli.IntFlag{Name: "tensors-limit", Usage: "limit tensor listing (0 = no limit)", Value: 50, Destination: &tensorLimit},
			&cli.IntFlag{Name: "vocab-limit", Usage: "limit vocab listing (0 = no limit)", Value: 50, Destination: &vocabLimit},
			&cli.StringFlag{Name: "tensor-filter", Usage: "substring filter for tensor listing", Destination: &tensorFilter},
			&cli.BoolFlag{Name: "tokenizer-json", Usage: "print raw tokenizer.json", Destination: &showTokenizerJS},
			&cli.BoolFlag{Name: "tokenizer-config", Usage: "print raw tokenizer_config.json", Destination: &showTokenizerCfg},
			&cli.BoolFlag{Name: "hf-config", Usage: "print raw config.json", Destination: &showHFConfig},
			&cli.BoolFlag{Name: "generation-config", Usage: "print raw generation_config.json", Destination: &showGenConfig},
			&cli.BoolFlag{Name: "vocab-json", Usage: "print raw vocab.json", Destination: &showVocabJSON},
			&cli.BoolFlag{Name: "merges", Usage: "print raw merges.txt", Destination: &showMerges},
			&cli.BoolFlag{Name: "modelinfo", Usage: "print raw modelinfo", Destination: &showModelInfo},
			&cli.BoolFlag{Name: "tensor-data", Usage: "print tensor data section bounds", Destination: &showTensorData},
		},
		Action: func(ctx context.Context, c *cli.Command) error {
			_ = ctx

			if showAll {
				showSections = true
				showTensors = true
				showQuant = true
				showTokenizer = true
				showChat = true
				showVocab = true
				showTokenizerJS = true
				showTokenizerCfg = true
				showHFConfig = true
				showGenConfig = true
				showVocabJSON = true
				showMerges = true
				showModelInfo = true
				showTensorData = true
				if tensorLimit == 50 {
					tensorLimit = 0
				}
				if vocabLimit == 50 {
					vocabLimit = 0
				}
			}

			stat, err := os.Stat(modelPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: stat model path %q: %v", modelPath, err), 1)
			}
			if stat.IsDir() || !strings.HasSuffix(strings.ToLower(modelPath), ".mcf") {
				return cli.Exit("error: mantle inspect only supports .mcf files", 1)
			}

			mf, err := mcf.Open(modelPath)
			if err != nil {
				return cli.Exit(fmt.Sprintf("error: open mcf: %v", err), 1)
			}
			defer func() { _ = mf.Close() }()

			fmt.Printf("MCF Inspect: %s\n", modelPath)
			fmt.Printf("File: %s (%s)\n", filepath.Base(modelPath), formatBytes(uint64(stat.Size())))
			printHeader(mf.Header)

			sections := mf.Sections
			sec := func(t mcf.SectionType) *mcf.MCFSection { return mf.Section(t) }
			getData := func(t mcf.SectionType) []byte { return mf.SectionData(sec(t)) }

			cfgBytes := getData(mcf.SectionHFConfigJSON)
			genBytes := getData(mcf.SectionHFGenerationConfigJSON)
			tokJSON := getData(mcf.SectionTokenizerJSON)
			tokCfg := getData(mcf.SectionTokenizerConfigJSON)
			vocabJSON := getData(mcf.SectionVocabJSON)
			mergesTXT := getData(mcf.SectionMergesTXT)
			modelInfo := getData(mcf.SectionModelInfo)
			tensorIndexBytes := getData(mcf.SectionTensorIndex)
			quantBytes := getData(mcf.SectionQuantInfo)

			cfg := parseHFConfig(cfgBytes)
			printParameters(cfg)

			tokSummary := parseTokenizerSummary(tokJSON, tokCfg)
			printTokenizerSummary(tokSummary)

			printTensorSummary(tensorIndexBytes, quantBytes)

			if showSections {
				printSectionDirectory(sections)
			}

			if showTensors {
				printTensorIndex(tensorIndexBytes, quantBytes, tensorFilter, tensorLimit, showQuant)
			}

			if showTokenizer {
				printTokenizerDetails(tokSummary)
			}

			if showChat {
				printChatTemplate(tokSummary.ChatTemplate)
			}

			if showVocab {
				printVocab(tokJSON, vocabJSON, vocabLimit)
			}

			if showHFConfig {
				printRawSection("HF Config (config.json)", cfgBytes)
			}
			if showGenConfig {
				printRawSection("Generation Config (generation_config.json)", genBytes)
			}
			if showTokenizerJS {
				printRawSection("Tokenizer JSON (tokenizer.json)", tokJSON)
			}
			if showTokenizerCfg {
				printRawSection("Tokenizer Config (tokenizer_config.json)", tokCfg)
			}
			if showVocabJSON {
				printRawSection("Vocab JSON (vocab.json)", vocabJSON)
			}
			if showMerges {
				printRawSection("Merges (merges.txt)", mergesTXT)
			}
			if showModelInfo {
				printRawSection("Model Info", modelInfo)
			}
			if showTensorData {
				if s := sec(mcf.SectionTensorData); s != nil {
					printSectionBounds("Tensor Data", s)
				}
			}

			return nil
		},
	}
}

type hfConfigSummary struct {
	ModelType     string
	TextModelType string
	Architectures []string
	HiddenSize    int
	Intermediate  int
	NumLayers     int
	HeadCount     int
	HeadDim       int
	KVHeads       int
	MaxPosition   int
	VocabSize     int
	RMSNormEps    float64
	RopeTheta     float64
	RopeType      string
	RopeFactor    float64
	RopeOrigCtx   int
	RopeLow       float64
	RopeHigh      float64
	RopeAttn      float64
}

type tokenizerSummary struct {
	TokenizerType string
	VocabSize     int
	MergeCount    int
	AddedTokens   int
	SpecialTokens int
	AddBOS        bool
	AddEOS        bool
	BOS           string
	EOS           string
	UNK           string
	PAD           string
	BOSID         int
	EOSID         int
	UNKID         int
	PADID         int
	ChatTemplate  string
}

func parseHFConfig(raw []byte) hfConfigSummary {
	if len(raw) == 0 {
		return hfConfigSummary{}
	}
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil {
		return hfConfigSummary{}
	}
	textCfg, _ := root["text_config"].(map[string]any)
	get := func(key string) any {
		if v, ok := root[key]; ok {
			return v
		}
		if textCfg != nil {
			if v, ok := textCfg[key]; ok {
				return v
			}
		}
		return nil
	}
	getText := func(key string) any {
		if textCfg == nil {
			return nil
		}
		return textCfg[key]
	}

	archs := []string{}
	if v, ok := root["architectures"].([]any); ok {
		for _, item := range v {
			if s, ok := item.(string); ok {
				archs = append(archs, s)
			}
		}
	}

	s := hfConfigSummary{
		ModelType:     getString(root, "model_type"),
		TextModelType: getStringAny(getText("model_type")),
		Architectures: archs,
		HiddenSize:    getIntAny(get("hidden_size")),
		Intermediate:  getIntAny(get("intermediate_size")),
		NumLayers:     getIntAny(get("num_hidden_layers")),
		HeadCount:     getIntAny(get("num_attention_heads")),
		HeadDim:       getIntAny(get("head_dim")),
		KVHeads:       getIntAny(get("num_key_value_heads")),
		MaxPosition:   getIntAny(get("max_position_embeddings")),
		VocabSize:     getIntAny(get("vocab_size")),
		RMSNormEps:    getFloatAny(get("rms_norm_eps")),
		RopeTheta:     getFloatAny(get("rope_theta")),
	}

	if s.RopeTheta == 0 {
		if rp, ok := get("rope_parameters").(map[string]any); ok {
			s.RopeTheta = getFloatAny(rp["rope_theta"])
		}
	}
	if rs, ok := get("rope_scaling").(map[string]any); ok {
		applyRopeScaling(&s, rs)
	}
	if rp, ok := get("rope_parameters").(map[string]any); ok {
		applyRopeScaling(&s, rp)
	}

	return s
}

func applyRopeScaling(s *hfConfigSummary, m map[string]any) {
	if s == nil || m == nil {
		return
	}
	if s.RopeType == "" {
		s.RopeType = getStringAny(m["rope_type"])
		if s.RopeType == "" {
			s.RopeType = getStringAny(m["type"])
		}
	}
	if s.RopeFactor == 0 {
		s.RopeFactor = getFloatAny(m["factor"])
	}
	if s.RopeOrigCtx == 0 {
		s.RopeOrigCtx = getIntAny(m["original_max_position_embeddings"])
	}
	if s.RopeLow == 0 {
		if v := getFloatAny(m["low_freq_factor"]); v != 0 {
			s.RopeLow = v
		}
	}
	if s.RopeHigh == 0 {
		if v := getFloatAny(m["high_freq_factor"]); v != 0 {
			s.RopeHigh = v
		}
	}
	if s.RopeAttn == 0 {
		if v := getFloatAny(m["attention_factor"]); v != 0 {
			s.RopeAttn = v
		}
	}
}

func parseTokenizerSummary(tokJSON, tokCfg []byte) tokenizerSummary {
	var out tokenizerSummary
	if len(tokJSON) == 0 {
		return out
	}
	var tj tokenizerJSON
	if err := json.Unmarshal(tokJSON, &tj); err == nil {
		out.TokenizerType = tj.Model.Type
		out.VocabSize = len(tj.Model.Vocab)
		out.MergeCount = len(tj.Model.Merges)
		out.AddedTokens = len(tj.AddedTokens)
		for _, t := range tj.AddedTokens {
			if t.Special {
				out.SpecialTokens++
			}
		}
	}

	var cfg tokenizerConfigJSON
	if len(tokCfg) > 0 {
		_ = json.Unmarshal(tokCfg, &cfg)
		out.ChatTemplate = cfg.ChatTemplate
		out.AddBOS = cfg.AddBOS
		out.AddEOS = cfg.AddEOS
		out.BOS = cfg.BOS
		out.EOS = cfg.EOS
		out.UNK = cfg.UNK
		out.PAD = cfg.PAD
		out.BOSID = derefInt(cfg.BOSID, -1)
		out.EOSID = derefInt(cfg.EOSID, -1)
		out.UNKID = derefInt(cfg.UNKID, -1)
		out.PADID = derefInt(cfg.PADID, -1)
	}

	if len(tokJSON) > 0 {
		if parsedCfg, err := tokenizer.ParseHFTokenizerConfigBytes(tokJSON, tokCfg); err == nil {
			out.AddBOS = parsedCfg.AddBOS
			out.AddEOS = parsedCfg.AddEOS
			out.BOSID = parsedCfg.BOSTokenID
			out.EOSID = parsedCfg.EOSTokenID
			out.PADID = parsedCfg.PADTokenID
			out.UNKID = parsedCfg.UNKTokenID
			if out.ChatTemplate == "" {
				out.ChatTemplate = parsedCfg.ChatTemplate
			}
		}
	}

	return out
}

func printHeader(h *mcf.MCFHeader) {
	if h == nil {
		return
	}
	flags := []string{}
	if h.Flags&mcf.FlagTensorDataAligned64 != 0 {
		flags = append(flags, "tensor_data_aligned64")
	}
	flagStr := "none"
	if len(flags) > 0 {
		flagStr = strings.Join(flags, ", ")
	}
	fmt.Printf("MCF Header: v%d.%d sections=%d header=%dB flags=%s\n",
		h.Major, h.Minor, h.SectionCount, h.HeaderSize, flagStr)
}

func printParameters(cfg hfConfigSummary) {
	section("Parameters")
	row("model_type", cfg.ModelType)
	if cfg.TextModelType != "" && cfg.TextModelType != cfg.ModelType {
		row("text_model_type", cfg.TextModelType)
	}
	if len(cfg.Architectures) > 0 {
		row("architectures", strings.Join(cfg.Architectures, ", "))
	}
	rowInt("hidden_size", cfg.HiddenSize)
	rowInt("intermediate_size", cfg.Intermediate)
	rowInt("num_layers", cfg.NumLayers)
	rowInt("num_attention_heads", cfg.HeadCount)
	rowInt("num_key_value_heads", cfg.KVHeads)
	rowInt("head_dim", cfg.HeadDim)
	rowInt("vocab_size", cfg.VocabSize)
	rowInt("max_position_embeddings", cfg.MaxPosition)
	rowFloat("rms_norm_eps", cfg.RMSNormEps)
	rowFloat("rope_theta", cfg.RopeTheta)
	if cfg.RopeType != "" || cfg.RopeFactor != 0 || cfg.RopeOrigCtx != 0 {
		low := cfg.RopeLow
		if low == 0 {
			low = 1
		}
		high := cfg.RopeHigh
		if high == 0 {
			high = 1
		}
		attn := cfg.RopeAttn
		if attn == 0 {
			attn = 1
		}
		row("rope", fmt.Sprintf("type=%s factor=%g orig_ctx=%d low=%g high=%g attn=%g",
			cfg.RopeType, cfg.RopeFactor, cfg.RopeOrigCtx, low, high, attn))
	}
}

func printTokenizerSummary(s tokenizerSummary) {
	section("Tokenizer Summary")
	row("tokenizer_type", s.TokenizerType)
	rowInt("vocab_size", s.VocabSize)
	rowInt("merges", s.MergeCount)
	rowInt("added_tokens", s.AddedTokens)
	rowInt("special_tokens", s.SpecialTokens)
	row("add_bos", fmt.Sprintf("%v", s.AddBOS))
	row("add_eos", fmt.Sprintf("%v", s.AddEOS))
	row("bos", formatTokenInfo(s.BOS, s.BOSID))
	row("eos", formatTokenInfo(s.EOS, s.EOSID))
	row("unk", formatTokenInfo(s.UNK, s.UNKID))
	row("pad", formatTokenInfo(s.PAD, s.PADID))
	if s.ChatTemplate != "" {
		row("chat_template", fmt.Sprintf("present (%d chars)", len(s.ChatTemplate)))
	} else {
		row("chat_template", "none")
	}
}

func printTokenizerDetails(s tokenizerSummary) {
	section("Tokenizer Details")
	row("tokenizer_type", s.TokenizerType)
	rowInt("vocab_size", s.VocabSize)
	rowInt("merges", s.MergeCount)
	rowInt("added_tokens", s.AddedTokens)
	rowInt("special_tokens", s.SpecialTokens)
	row("add_bos", fmt.Sprintf("%v", s.AddBOS))
	row("add_eos", fmt.Sprintf("%v", s.AddEOS))
	row("bos", formatTokenInfo(s.BOS, s.BOSID))
	row("eos", formatTokenInfo(s.EOS, s.EOSID))
	row("unk", formatTokenInfo(s.UNK, s.UNKID))
	row("pad", formatTokenInfo(s.PAD, s.PADID))
}

func printChatTemplate(tpl string) {
	section("Chat Template")
	if strings.TrimSpace(tpl) == "" {
		fmt.Println("(none)")
		return
	}
	fmt.Println(tpl)
}

func printSectionDirectory(sections []mcf.MCFSection) {
	section("Sections")
	for _, s := range sections {
		name := sectionTypeName(mcf.SectionType(s.Type))
		fmt.Printf("%-28s v%-2d off=%-10d size=%s\n", name, s.Version, s.Offset, formatBytes(s.Size))
	}
}

func printSectionBounds(name string, s *mcf.MCFSection) {
	section(name)
	fmt.Printf("off=%d size=%s\n", s.Offset, formatBytes(s.Size))
}

func printTensorSummary(indexBytes, quantBytes []byte) {
	section("Tensor Summary")
	if len(indexBytes) == 0 {
		fmt.Println("(no tensor index section)")
		return
	}
	idx, err := mcf.ParseTensorIndexSection(indexBytes)
	if err != nil {
		fmt.Printf("(tensor index parse error: %v)\n", err)
		return
	}

	count := idx.Count()
	rowInt("tensors", count)

	dtypeCounts := map[mcf.TensorDType]int{}
	dtypeBytes := map[mcf.TensorDType]uint64{}
	var total uint64
	for i := range count {
		entry, err := idx.Entry(i)
		if err != nil {
			continue
		}
		dtypeCounts[entry.DType]++
		dtypeBytes[entry.DType] += entry.DataSize
		total += entry.DataSize
	}
	row("data_size", formatBytes(total))

	keys := make([]mcf.TensorDType, 0, len(dtypeCounts))
	for k := range dtypeCounts {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	for _, k := range keys {
		row(fmt.Sprintf("dtype_%s", dtypeName(k)), fmt.Sprintf("%d (%s)", dtypeCounts[k], formatBytes(dtypeBytes[k])))
	}

	if len(quantBytes) > 0 {
		qi, err := mcf.ParseQuantInfoSection(quantBytes)
		if err == nil {
			rowInt("quant_records", qi.Count())
		}
	}
}

func printTensorIndex(indexBytes, quantBytes []byte, filter string, limit int, showQuant bool) {
	section("Tensor Index")
	if len(indexBytes) == 0 {
		fmt.Println("(no tensor index section)")
		return
	}
	idx, err := mcf.ParseTensorIndexSection(indexBytes)
	if err != nil {
		fmt.Printf("(tensor index parse error: %v)\n", err)
		return
	}

	quantMap := map[uint32]mcf.QuantRecord{}
	if len(quantBytes) > 0 {
		if qi, err := mcf.ParseQuantInfoSection(quantBytes); err == nil {
			for i := 0; i < qi.Count(); i++ {
				r, _ := qi.Record(i)
				quantMap[r.TensorIndex] = r
			}
		}
	}

	count := idx.Count()
	printed := 0
	for i := range count {
		name, err := idx.Name(i)
		if err != nil {
			continue
		}
		if filter != "" && !strings.Contains(name, filter) {
			continue
		}
		entry, err := idx.Entry(i)
		if err != nil {
			continue
		}
		shape, _ := idx.Shape(i)
		line := fmt.Sprintf("%s  dtype=%s shape=%s size=%s", name, dtypeName(entry.DType), formatShape(shape), formatBytes(entry.DataSize))
		if showQuant {
			if r, ok := quantMap[uint32(i)]; ok {
				line += fmt.Sprintf(" quant=%s block=%d super=%d clip=[%g,%g]", dtypeName(mcf.TensorDType(r.Method)), r.BlockSize, r.SuperSize, r.MinClip, r.MaxClip)
			}
		}
		fmt.Println(line)
		printed++
		if limit > 0 && printed >= limit {
			break
		}
	}
	if limit > 0 && printed < count {
		fmt.Printf("... (%d shown of %d)\n", printed, count)
	}
}

func printVocab(tokJSON, vocabJSON []byte, limit int) {
	section("Vocab")
	entries := map[string]int{}
	if len(vocabJSON) > 0 {
		_ = json.Unmarshal(vocabJSON, &entries)
	} else if len(tokJSON) > 0 {
		var tj tokenizerJSON
		if err := json.Unmarshal(tokJSON, &tj); err == nil {
			entries = tj.Model.Vocab
		}
	}
	if len(entries) == 0 {
		fmt.Println("(no vocab section)")
		return
	}
	ids := make([]int, 0, len(entries))
	inv := make(map[int]string, len(entries))
	for tok, id := range entries {
		inv[id] = tok
		ids = append(ids, id)
	}
	sort.Ints(ids)
	count := len(ids)
	shown := 0
	for _, id := range ids {
		fmt.Printf("%6d  %s\n", id, inv[id])
		shown++
		if limit > 0 && shown >= limit {
			break
		}
	}
	if limit > 0 && shown < count {
		fmt.Printf("... (%d shown of %d)\n", shown, count)
	}
}

func printRawSection(name string, data []byte) {
	section(name)
	if len(data) == 0 {
		fmt.Println("(missing)")
		return
	}
	fmt.Println(string(data))
}

func section(title string) {
	line := strings.Repeat("-", len(title)+8)
	fmt.Printf("\n%s\n--- %s ---\n%s\n", line, title, line)
}

func row(label, value string) {
	if value == "" {
		return
	}
	fmt.Printf("%-24s %s\n", label+":", value)
}

func rowInt(label string, v int) {
	if v == 0 {
		return
	}
	row(label, fmt.Sprintf("%d", v))
}

func rowFloat(label string, v float64) {
	if v == 0 {
		return
	}
	row(label, fmt.Sprintf("%g", v))
}

func formatShape(shape []uint64) string {
	if len(shape) == 0 {
		return "[]"
	}
	parts := make([]string, len(shape))
	for i, v := range shape {
		parts[i] = fmt.Sprintf("%d", v)
	}
	return "[" + strings.Join(parts, " ") + "]"
}

func formatTokenInfo(tok string, id int) string {
	if tok == "" && id < 0 {
		return "-"
	}
	if tok == "" {
		return fmt.Sprintf("id=%d", id)
	}
	if id < 0 {
		return fmt.Sprintf("%q", tok)
	}
	return fmt.Sprintf("%q (id=%d)", tok, id)
}

func formatBytes(b uint64) string {
	const (
		kb = 1024
		mb = 1024 * kb
		gb = 1024 * mb
		tb = 1024 * gb
	)
	switch {
	case b >= tb:
		return fmt.Sprintf("%.2f TiB", float64(b)/float64(tb))
	case b >= gb:
		return fmt.Sprintf("%.2f GiB", float64(b)/float64(gb))
	case b >= mb:
		return fmt.Sprintf("%.2f MiB", float64(b)/float64(mb))
	case b >= kb:
		return fmt.Sprintf("%.2f KiB", float64(b)/float64(kb))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func sectionTypeName(t mcf.SectionType) string {
	switch t {
	case mcf.SectionModelInfo:
		return "ModelInfo"
	case mcf.SectionQuantInfo:
		return "QuantInfo"
	case mcf.SectionTensorIndex:
		return "TensorIndex"
	case mcf.SectionTensorData:
		return "TensorData"
	case mcf.SectionHFConfigJSON:
		return "HFConfigJSON"
	case mcf.SectionHFGenerationConfigJSON:
		return "HFGenerationConfigJSON"
	case mcf.SectionTokenizerJSON:
		return "TokenizerJSON"
	case mcf.SectionTokenizerConfigJSON:
		return "TokenizerConfigJSON"
	case mcf.SectionVocabJSON:
		return "VocabJSON"
	case mcf.SectionMergesTXT:
		return "MergesTXT"
	default:
		return fmt.Sprintf("Section0x%04x", uint32(t))
	}
}

func dtypeName(dt mcf.TensorDType) string {
	switch dt {
	case mcf.DTypeF32:
		return "f32"
	case mcf.DTypeF16:
		return "f16"
	case mcf.DTypeBF16:
		return "bf16"
	case mcf.DTypeF64:
		return "f64"
	case mcf.DTypeI8:
		return "i8"
	case mcf.DTypeU8:
		return "u8"
	case mcf.DTypeI16:
		return "i16"
	case mcf.DTypeU16:
		return "u16"
	case mcf.DTypeI32:
		return "i32"
	case mcf.DTypeU32:
		return "u32"
	case mcf.DTypeI64:
		return "i64"
	case mcf.DTypeU64:
		return "u64"
	case mcf.DTypeInt8:
		return "int8"
	case mcf.DTypeInt4:
		return "int4"
	case mcf.DTypeQ8:
		return "q8"
	case mcf.DTypeQ4:
		return "q4"
	case mcf.DTypeK6:
		return "k6"
	case mcf.DTypeK4:
		return "k4"
	case mcf.DTypeK3:
		return "k3"
	case mcf.DTypeK2:
		return "k2"
	default:
		return fmt.Sprintf("dtype_%d", dt)
	}
}

func getString(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	if v, ok := m[key]; ok {
		return getStringAny(v)
	}
	return ""
}

func getStringAny(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func getIntAny(v any) int {
	if v == nil {
		return 0
	}
	switch t := v.(type) {
	case int:
		return t
	case int64:
		return int(t)
	case float64:
		return int(t)
	case json.Number:
		i, _ := t.Int64()
		return int(i)
	default:
		return 0
	}
}

func getFloatAny(v any) float64 {
	if v == nil {
		return 0
	}
	switch t := v.(type) {
	case float64:
		return t
	case float32:
		return float64(t)
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case json.Number:
		f, _ := t.Float64()
		return f
	default:
		return 0
	}
}

func derefInt(v *int, def int) int {
	if v == nil {
		return def
	}
	return *v
}
