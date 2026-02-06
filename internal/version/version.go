package version

import "time"

var (
	// Version is the release version (set via -ldflags).
	Version = ""
	// Commit is the git commit hash (set via -ldflags).
	Commit = ""
	// BuildTime is the build timestamp (set via -ldflags).
	BuildTime = ""
)

type Info struct {
	Version   string
	Commit    string
	BuildTime string
}

func Resolve() Info {
	resolved := Info{
		Version:   Version,
		Commit:    Commit,
		BuildTime: BuildTime,
	}

	if resolved.Version == "" {
		if resolved.BuildTime != "" {
			resolved.Version = resolved.BuildTime
		} else {
			resolved.Version = time.Now().UTC().Format("20060102T150405Z")
		}
	}

	return resolved
}

func String() string {
	info := Resolve()
	if info.Commit == "" {
		return info.Version
	}
	return info.Version + " (" + shortCommit(info.Commit) + ")"
}

func shortCommit(commit string) string {
	if len(commit) <= 12 {
		return commit
	}
	return commit[:12]
}
