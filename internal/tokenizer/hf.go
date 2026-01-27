package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

type HFTokenizer struct {
	encoder      map[string]int
	decoder      []string
	bpeRanks     map[Pair]int
	cache        map[string][]string
	byteEncoder  map[byte]string
	byteDecoder  map[string]byte
	pattern      *regexp.Regexp
	addBOS       bool
	addEOS       bool
	bosID        int
	eosID        int
	unkID        int
	ignoreMerges bool
	special      []string
}

type hfTokenizerJSON struct {
	Model struct {
		Type         string         `json:"type"`
		Vocab        map[string]int `json:"vocab"`
		Merges       []any          `json:"merges"`
		IgnoreMerges bool           `json:"ignore_merges"`
		UnkToken     string         `json:"unk_token"`
	} `json:"model"`
	PreTokenizer struct {
		Type          string `json:"type"`
		Pretokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
	PostProcessor struct {
		Type       string `json:"type"`
		Processors []struct {
			Type          string `json:"type"`
			SpecialTokens map[string]struct {
				IDs []int `json:"ids"`
			} `json:"special_tokens"`
		} `json:"processors"`
	} `json:"post_processor"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

type hfTokenizerConfig struct {
	AddBOS       bool   `json:"add_bos_token"`
	AddEOS       bool   `json:"add_eos_token"`
	BOS          string `json:"bos_token"`
	EOS          string `json:"eos_token"`
	PAD          string `json:"pad_token"`
	UNK          string `json:"unk_token"`
	BOSID        *int   `json:"bos_token_id"`
	EOSID        *int   `json:"eos_token_id"`
	PADID        *int   `json:"pad_token_id"`
	UNKID        *int   `json:"unk_token_id"`
	ChatTemplate string `json:"chat_template"`
}

func LoadHFTokenizer(tokJSON, tokConfig string) (*HFTokenizer, error) {
	data, err := os.ReadFile(tokJSON)
	if err != nil {
		return nil, err
	}
	var cfg []byte
	if tokConfig != "" {
		if raw, err := os.ReadFile(tokConfig); err == nil {
			cfg = raw
		}
	}
	return LoadHFTokenizerBytes(data, cfg)
}

func parseHFTokenizerJSON(tokJSON []byte) (hfTokenizerJSON, map[string]int, error) {
	var tj hfTokenizerJSON
	if err := json.Unmarshal(tokJSON, &tj); err != nil {
		return hfTokenizerJSON{}, nil, err
	}

	encoder := make(map[string]int, len(tj.Model.Vocab)+len(tj.AddedTokens))
	for tok, id := range tj.Model.Vocab {
		encoder[tok] = id
	}
	for _, at := range tj.AddedTokens {
		encoder[at.Content] = at.ID
	}
	return tj, encoder, nil
}

func bosIDFromTemplateProcessing(tj hfTokenizerJSON) int {
	best := -1
	for _, proc := range tj.PostProcessor.Processors {
		if proc.Type != "TemplateProcessing" {
			continue
		}
		for _, spec := range proc.SpecialTokens {
			if len(spec.IDs) == 0 {
				continue
			}
			id := spec.IDs[0]
			if id < 0 {
				continue
			}
			if best < 0 || id < best {
				best = id
			}
		}
	}
	return best
}

func LoadHFTokenizerBytes(tokJSON []byte, tokConfig []byte) (*HFTokenizer, error) {
	tj, encoder, err := parseHFTokenizerJSON(tokJSON)
	if err != nil {
		return nil, err
	}
	if strings.ToUpper(tj.Model.Type) != "BPE" {
		return nil, fmt.Errorf("unsupported tokenizer model: %s", tj.Model.Type)
	}

	maxID := -1
	for _, id := range encoder {
		if id > maxID {
			maxID = id
		}
	}
	decoder := make([]string, maxID+1)
	for tok, id := range encoder {
		decoder[id] = tok
	}

	bpeRanks := make(map[Pair]int, len(tj.Model.Merges))
	rank := 0
	for _, raw := range tj.Model.Merges {
		line := ""
		switch v := raw.(type) {
		case string:
			line = v
		case []any:
			if len(v) == 2 {
				a, aok := v[0].(string)
				b, bok := v[1].(string)
				if aok && bok {
					line = a + " " + b
				}
			}
		}
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			continue
		}
		p := Pair{A: parts[0], B: parts[1]}
		if _, ok := bpeRanks[p]; !ok {
			bpeRanks[p] = rank
			rank++
		}
	}

	byteEncoder, byteDecoder := bytesToUnicode()
	pat := buildHFPattern(tj.PreTokenizer)

	var cfg hfTokenizerConfig
	if len(tokConfig) > 0 {
		_ = json.Unmarshal(tokConfig, &cfg)
	}

	addBOS := cfg.AddBOS
	addEOS := cfg.AddEOS
	bosID := -1
	eosID := -1
	if cfg.BOS != "" {
		if id, ok := encoder[cfg.BOS]; ok {
			bosID = id
		}
	}
	if cfg.EOS != "" {
		if id, ok := encoder[cfg.EOS]; ok {
			eosID = id
		}
	}
	if cfg.BOSID != nil && *cfg.BOSID >= 0 {
		bosID = *cfg.BOSID
	}
	if cfg.EOSID != nil && *cfg.EOSID >= 0 {
		eosID = *cfg.EOSID
	}
	// If TemplateProcessing defines a BOS token, use it deterministically.
	if ppBosID := bosIDFromTemplateProcessing(tj); ppBosID >= 0 {
		bosID = ppBosID
		addBOS = true
	}

	unkID := -1
	if tj.Model.UnkToken != "" {
		if id, ok := encoder[tj.Model.UnkToken]; ok {
			unkID = id
		}
	}

	tok := &HFTokenizer{
		encoder:      encoder,
		decoder:      decoder,
		bpeRanks:     bpeRanks,
		cache:        make(map[string][]string),
		byteEncoder:  byteEncoder,
		byteDecoder:  byteDecoder,
		pattern:      pat,
		addBOS:       addBOS,
		addEOS:       addEOS,
		bosID:        bosID,
		eosID:        eosID,
		unkID:        unkID,
		ignoreMerges: tj.Model.IgnoreMerges,
		special:      collectSpecials(decoder),
	}
	return tok, nil
}

// ParseHFTokenizerConfigBytes extracts useful runtime settings (like chat_template)
// from tokenizer_config.json and tokenizer.json.
func ParseHFTokenizerConfigBytes(tokJSON []byte, tokConfig []byte) (TokenizerConfig, error) {
	tj, encoder, err := parseHFTokenizerJSON(tokJSON)
	if err != nil {
		return TokenizerConfig{}, err
	}
	if strings.ToUpper(tj.Model.Type) != "BPE" {
		return TokenizerConfig{}, fmt.Errorf("unsupported tokenizer model: %s", tj.Model.Type)
	}

	var cfg hfTokenizerConfig
	if len(tokConfig) > 0 {
		_ = json.Unmarshal(tokConfig, &cfg)
	}

	out := TokenizerConfig{
		AddBOS:       cfg.AddBOS,
		AddEOS:       cfg.AddEOS,
		ChatTemplate: cfg.ChatTemplate,
		BOSTokenID:   -1,
		EOSTokenID:   -1,
		PADTokenID:   -1,
		UNKTokenID:   -1,
	}

	resolveID := func(tok string, idPtr *int) int {
		if tok != "" {
			if id, ok := encoder[tok]; ok {
				return id
			}
		}
		if idPtr != nil && *idPtr >= 0 {
			return *idPtr
		}
		return -1
	}

	out.BOSTokenID = resolveID(cfg.BOS, cfg.BOSID)
	out.EOSTokenID = resolveID(cfg.EOS, cfg.EOSID)
	out.PADTokenID = resolveID(cfg.PAD, cfg.PADID)
	out.UNKTokenID = resolveID(cfg.UNK, cfg.UNKID)

	if ppBosID := bosIDFromTemplateProcessing(tj); ppBosID >= 0 {
		out.BOSTokenID = ppBosID
		out.AddBOS = true
	}

	return out, nil
}

func (t *HFTokenizer) Encode(text string) ([]int, error) {
	var ids []int
	if t.addBOS && t.bosID >= 0 {
		ids = append(ids, t.bosID)
	}
	parts := splitSpecials(text, t.special)
	for _, part := range parts {
		if part.isSpecial {
			id, ok := t.encoder[part.text]
			if !ok {
				return nil, fmt.Errorf("unknown special token: %q", part.text)
			}
			ids = append(ids, id)
			continue
		}
		for _, token := range t.pattern.FindAllString(part.text, -1) {
			encoded := t.byteEncode(token)
			bpeTokens := t.bpe(encoded)
			for _, bpeTok := range bpeTokens {
				id, ok := t.encoder[bpeTok]
				if !ok {
					if t.unkID >= 0 {
						ids = append(ids, t.unkID)
						continue
					}
					return nil, fmt.Errorf("unknown token: %q", bpeTok)
				}
				ids = append(ids, id)
			}
		}
	}
	if t.addEOS && t.eosID >= 0 {
		ids = append(ids, t.eosID)
	}
	return ids, nil
}

func (t *HFTokenizer) Decode(ids []int) (string, error) {
	var b []byte
	for _, id := range ids {
		if id < 0 || id >= len(t.decoder) {
			return "", fmt.Errorf("token id out of range: %d", id)
		}
		token := t.decoder[id]
		if isSpecialToken(token) {
			b = append(b, token...)
			continue
		}
		for _, r := range token {
			if by, ok := t.byteDecoder[string(r)]; ok {
				b = append(b, by)
			} else {
				b = append(b, string(r)...)
			}
		}
	}
	return string(b), nil
}

func (t *HFTokenizer) BOSID() int   { return t.bosID }
func (t *HFTokenizer) EOSID() int   { return t.eosID }
func (t *HFTokenizer) AddBOS() bool { return t.addBOS }
func (t *HFTokenizer) AddEOS() bool { return t.addEOS }
func (t *HFTokenizer) TokenString(id int) string {
	if id < 0 || id >= len(t.decoder) {
		return ""
	}
	return t.decoder[id]
}

func (t *HFTokenizer) byteEncode(s string) string {
	var b strings.Builder
	for _, by := range []byte(s) {
		b.WriteString(t.byteEncoder[by])
	}
	return b.String()
}

func (t *HFTokenizer) bpe(token string) []string {
	if v, ok := t.cache[token]; ok {
		return v
	}
	if t.ignoreMerges {
		if _, ok := t.encoder[token]; ok {
			out := []string{token}
			t.cache[token] = out
			return out
		}
	}
	word := splitRunes(token)
	pairs := getPairs(word)
	for len(pairs) > 0 {
		bestRank := int(^uint(0) >> 1)
		bestPair := Pair{}
		found := false
		for p := range pairs {
			if rank, ok := t.bpeRanks[p]; ok {
				if rank < bestRank {
					bestRank = rank
					bestPair = p
					found = true
				}
			}
		}
		if !found {
			break
		}
		word = mergePair(word, bestPair)
		if len(word) == 1 {
			break
		}
		pairs = getPairs(word)
	}
	t.cache[token] = word
	return word
}

func buildHFPattern(pre struct {
	Type          string `json:"type"`
	Pretokenizers []struct {
		Type    string `json:"type"`
		Pattern struct {
			Regex string `json:"Regex"`
		} `json:"pattern"`
	} `json:"pretokenizers"`
}) *regexp.Regexp {
	// Default to GPT2-ish regex.
	pat := `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`
	if pre.Type == "Sequence" {
		for _, p := range pre.Pretokenizers {
			if p.Type == "Split" && p.Pattern.Regex != "" {
				pat = p.Pattern.Regex
				break
			}
		}
	}
	// Some HF tokenizers ship PCRE-style regexes (lookahead, atomic groups,
	// anchors like \A/\G/\z) that Go's regexp package does not support.
	// Replace with a llama.cpp-compatible variant when we detect them.
	if strings.Contains(pat, "(?!\\S)") ||
		strings.Contains(pat, "(?i:") ||
		strings.Contains(pat, "(?=") ||
		strings.Contains(pat, "(?<") ||
		strings.Contains(pat, "(?>") ||
		strings.Contains(pat, `\G`) ||
		strings.Contains(pat, `\A`) ||
		strings.Contains(pat, `\z`) {
		pat = `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`
	}
	return regexp.MustCompile(pat)
}

func (t *HFTokenizer) Decoder() []string {
	return t.decoder
}
