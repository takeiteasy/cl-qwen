(asdf:defsystem "cl-qwen/gguf"
  :description "Validated, mapped GGUF reader"
  :license "MIT"
  :depends-on ("cffi" "babel")
  :serial t
  :components ((:file "src/gguf")))

(asdf:defsystem "cl-qwen"
  :description "Common Lisp Qwen3 inference with narrow numerical primitives"
  :license "MIT"
  :version "0.1.0"
  :depends-on ("cl-qwen/gguf" "trivial-simd/blas" "bordeaux-threads" "cl-ppcre-unicode")
  :serial t
  :components ((:file "src/package") (:file "src/numeric")
               (:file "src/tokenizer") (:file "src/model") (:file "src/session") (:file "src/kernel-numeric"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-qwen/tests"))))

(asdf:defsystem "cl-qwen/tests"
  :depends-on ("cl-qwen" "fiveam")
  :serial t
  :components ((:file "tests/suite"))
  :perform (asdf:test-op (op component)
             (declare (ignore op component))
             (unless (uiop:symbol-call :cl-qwen/tests :run-tests)
               (error "cl-qwen tests failed"))))
