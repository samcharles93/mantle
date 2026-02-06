package tokenizer

import "strings"

// Pair represents a pair of BPE tokens.
type Pair struct {
	A string
	B string
}

type textPart struct {
	text      string
	isSpecial bool
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
