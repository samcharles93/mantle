package tokenizer

import (
	"fmt"
	"regexp"
	"strings"
)

type Pair struct {
	A string
	B string
}

type GPT2Tokenizer struct {
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

func NewGPT2(tokens []string, merges []string, pre string, addBOS, addEOS bool, bosID, eosID, unkID int) (*GPT2Tokenizer, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty token list")
	}
	encoder := make(map[string]int, len(tokens))
	for i, t := range tokens {
		encoder[t] = i
	}
	decoder := append([]string(nil), tokens...)

	bpeRanks := make(map[Pair]int, len(merges))
	rank := 0
	for _, line := range merges {
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
	// Go regexp does not support lookahead, so we collapse the trailing
	// whitespace branch into a plain \s+ match.
	pat := regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)
	ignoreMerges := false
	if pre == "lfm2" || pre == "llama3" || pre == "llama-v3" || pre == "llama-bpe" || pre == "falcon3" || pre == "falcon-h1" || pre == "pixtral" || pre == "midm-2.0" {
		// Llama 3 style pre-tokenizer (lookahead removed for Go regexp compatibility).
		pat = regexp.MustCompile(`(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`)
		ignoreMerges = true
	}

	return &GPT2Tokenizer{
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
		ignoreMerges: ignoreMerges,
		special:      collectSpecials(tokens),
	}, nil
}

func (t *GPT2Tokenizer) Encode(text string) ([]int, error) {
	var ids []int
	if t.addBOS {
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
	if t.addEOS {
		ids = append(ids, t.eosID)
	}
	return ids, nil
}

func (t *GPT2Tokenizer) Decode(ids []int) (string, error) {
	var b []byte
	for _, id := range ids {
		if id < 0 || id >= len(t.decoder) {
			return "", fmt.Errorf("token id out of range: %d", id)
		}
		token := t.decoder[id]
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

func (t *GPT2Tokenizer) BOSID() int   { return t.bosID }
func (t *GPT2Tokenizer) EOSID() int   { return t.eosID }
func (t *GPT2Tokenizer) AddBOS() bool { return t.addBOS }
func (t *GPT2Tokenizer) AddEOS() bool { return t.addEOS }
func (t *GPT2Tokenizer) TokenString(id int) string {
	if id < 0 || id >= len(t.decoder) {
		return ""
	}
	return t.decoder[id]
}

func (t *GPT2Tokenizer) byteEncode(s string) string {
	var b strings.Builder
	for _, by := range []byte(s) {
		b.WriteString(t.byteEncoder[by])
	}
	return b.String()
}

func (t *GPT2Tokenizer) bpe(token string) []string {
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

func splitRunes(s string) []string {
	out := make([]string, 0, len(s))
	for _, r := range s {
		out = append(out, string(r))
	}
	return out
}

func getPairs(word []string) map[Pair]struct{} {
	pairs := make(map[Pair]struct{})
	if len(word) < 2 {
		return pairs
	}
	prev := word[0]
	for _, w := range word[1:] {
		pairs[Pair{A: prev, B: w}] = struct{}{}
		prev = w
	}
	return pairs
}

func mergePair(word []string, pair Pair) []string {
	var out []string
	for i := 0; i < len(word); i++ {
		if i < len(word)-1 && word[i] == pair.A && word[i+1] == pair.B {
			out = append(out, word[i]+word[i+1])
			i++
			continue
		}
		out = append(out, word[i])
	}
	return out
}

type textPart struct {
	text      string
	isSpecial bool
}

func collectSpecials(tokens []string) []string {
	out := make([]string, 0, 32)
	for _, t := range tokens {
		if isSpecialToken(t) {
			out = append(out, t)
		}
	}
	// longest-match first
	for i := 1; i < len(out); i++ {
		j := i
		for j > 0 && len(out[j]) > len(out[j-1]) {
			out[j], out[j-1] = out[j-1], out[j]
			j--
		}
	}
	return out
}

func isSpecialToken(s string) bool {
	if len(s) < 4 {
		return false
	}
	return strings.HasPrefix(s, "<|") && strings.HasSuffix(s, "|>")
}

func splitSpecials(text string, specials []string) []textPart {
	if len(specials) == 0 || !strings.Contains(text, "<|") {
		return []textPart{{text: text, isSpecial: false}}
	}
	var parts []textPart
	var buf strings.Builder
	for i := 0; i < len(text); {
		match := ""
		for _, sp := range specials {
			if len(sp) == 0 || i+len(sp) > len(text) {
				continue
			}
			if text[i:i+len(sp)] == sp {
				match = sp
				break
			}
		}
		if match != "" {
			if buf.Len() > 0 {
				parts = append(parts, textPart{text: buf.String(), isSpecial: false})
				buf.Reset()
			}
			parts = append(parts, textPart{text: match, isSpecial: true})
			i += len(match)
			continue
		}
		buf.WriteByte(text[i])
		i++
	}
	if buf.Len() > 0 {
		parts = append(parts, textPart{text: buf.String(), isSpecial: false})
	}
	return parts
}

// bytesToUnicode maps bytes to unicode strings to make BPE reversible.
func bytesToUnicode() (map[byte]string, map[string]byte) {
	var bs []int
	for i := int('!'); i <= int('~'); i++ {
		bs = append(bs, i)
	}
	for i := int('¡'); i <= int('¬'); i++ {
		bs = append(bs, i)
	}
	for i := int('®'); i <= int('ÿ'); i++ {
		bs = append(bs, i)
	}

	cs := make([]int, len(bs))
	copy(cs, bs)
	n := 0
	for b := 0; b < 256; b++ {
		found := false
		for _, v := range bs {
			if v == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}

	byteEncoder := make(map[byte]string, len(bs))
	byteDecoder := make(map[string]byte, len(bs))
	for i := 0; i < len(bs); i++ {
		b := byte(bs[i])
		r := rune(cs[i])
		s := string(r)
		byteEncoder[b] = s
		byteDecoder[s] = b
	}
	return byteEncoder, byteDecoder
}
