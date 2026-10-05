(load (merge-pathnames "../tools/bootstrap.lisp" *load-truename*))
(let ((model (cl-qwen:load-model (first (uiop:command-line-arguments)))))
  (unwind-protect
       (let ((session (cl-qwen:make-session model :context-size 256)))
         (multiple-value-bind (ids text)
             (cl-qwen:generate session "What is the capital of France?" :chat t :max-tokens 16)
           (format t "~A~%Tokens: ~S~%" text ids)))
    (cl-qwen:close-model model)))
