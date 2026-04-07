package ux

import (
	"fmt"
	"strings"
)

// Table represents a simple CLI table for formatting data.
type Table struct {
	Header []string
	Rows   [][]string
}

// NewTable creates a new Table.
func NewTable(header ...string) *Table {
	return &Table{
		Header: header,
	}
}

// AddRow adds a row to the table.
func (t *Table) AddRow(row ...string) {
	t.Rows = append(t.Rows, row)
}

// String returns the formatted table as a string.
func (t *Table) String() string {
	if len(t.Header) == 0 && len(t.Rows) == 0 {
		return ""
	}

	numCols := len(t.Header)
	for _, row := range t.Rows {
		if len(row) > numCols {
			numCols = len(row)
		}
	}

	colWidths := make([]int, numCols)
	for i, h := range t.Header {
		colWidths[i] = len(h)
	}

	for _, row := range t.Rows {
		for i, cell := range row {
			if len(cell) > colWidths[i] {
				colWidths[i] = len(cell)
			}
		}
	}

	var sb strings.Builder

	// Header
	if len(t.Header) > 0 {
		for i, h := range t.Header {
			fmt.Fprintf(&sb, "%-*s  ", colWidths[i], h)
		}
		sb.WriteString("\n")

		// Separator
		for i := range t.Header {
			sb.WriteString(strings.Repeat("-", colWidths[i]))
			sb.WriteString("  ")
		}
		sb.WriteString("\n")
	}

	// Rows
	for _, row := range t.Rows {
		for i := 0; i < numCols; i++ {
			cell := ""
			if i < len(row) {
				cell = row[i]
			}
			fmt.Fprintf(&sb, "%-*s  ", colWidths[i], cell)
		}
		sb.WriteString("\n")
	}

	return sb.String()
}
