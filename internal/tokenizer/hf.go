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
	AddBOS bool   `json:"add_bos_token"`
	AddEOS bool   `json:"add_eos_token"`
	BOS    string `json:"bos_token"`
	EOS    string `json:"eos_token"`
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

func LoadHFTokenizerBytes(tokJSON []byte, tokConfig []byte) (*HFTokenizer, error) {
	var tj hfTokenizerJSON
	if err := json.Unmarshal(tokJSON, &tj); err != nil {
		return nil, err
	}
	if strings.ToUpper(tj.Model.Type) != "BPE" {
		return nil, fmt.Errorf("unsupported tokenizer model: %s", tj.Model.Type)
	}

	encoder := make(map[string]int, len(tj.Model.Vocab))
	maxID := -1
	for tok, id := range tj.Model.Vocab {
		encoder[tok] = id
		if id > maxID {
			maxID = id
		}
	}
	for _, at := range tj.AddedTokens {
		if at.ID > maxID {
			maxID = at.ID
		}
	}
	decoder := make([]string, maxID+1)
	for tok, id := range tj.Model.Vocab {
		decoder[id] = tok
	}
	for _, at := range tj.AddedTokens {
		decoder[at.ID] = at.Content
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
	// If TemplateProcessing defines a BOS token, use it.
	for _, proc := range tj.PostProcessor.Processors {
		if proc.Type == "TemplateProcessing" {
			for _, spec := range proc.SpecialTokens {
				if len(spec.IDs) > 0 {
					bosID = spec.IDs[0]
					addBOS = true
					break
				}
			}
		}
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
	// LFM2 uses a Llama3-style regex with lookahead not supported by Go. Replace with llama.cpp variant.
	if strings.Contains(pat, "(?!\\S)") || strings.Contains(pat, "(?i:") {
		pat = `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`
	}
	return regexp.MustCompile(pat)
}

func (t *HFTokenizer) Decoder() []string {
	return t.decoder
}
