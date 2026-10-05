(load (merge-pathnames "../tools/bootstrap.lisp" *load-truename*))
(when (find-package :ql) (uiop:symbol-call :ql :quickload "cl-qwen/tests" :silent t))
(asdf:test-system "cl-qwen")
