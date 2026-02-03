package api

import "errors"

var ErrInvalidRequest = errors.New("invalid_request")

type invalidRequestError struct {
	msg string
}

func (e invalidRequestError) Error() string {
	return e.msg
}

func (e invalidRequestError) Unwrap() error {
	return ErrInvalidRequest
}

func newInvalidRequest(msg string) error {
	return invalidRequestError{msg: msg}
}
