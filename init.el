;; ===================== Emacs 初始化配置 =====================
;; 支持 C / C++ 开发、自动补全、VSCode 风格主题等

;; --------------------- 包管理器设置 ---------------------
(require 'package)
(setq package-archives
      '(("melpa" . "https://melpa.org/packages/")
        ("gnu"   . "https://elpa.gnu.org/packages/")))
(package-initialize)

;; 2. 如果没有安装 use-package，就刷新并安装它
(unless (package-installed-p 'use-package)
  (package-refresh-contents)
  (package-install 'use-package))

;; 3. 加载 use-package，并令它自动安装后续声明的包
(require 'use-package)
(setq use-package-always-ensure t)

(defvar my-packages
  '(company material-theme doom-themes
    markdown-mode git-timemachine magit flycheck lsp-ui lsp-mode typescript-mode))
(custom-set-variables
 '(package-selected-packages
   '(company material-theme doom-themes markdown-mode git-timemachine
     magit flycheck lsp-ui lsp-mode typescript-mode)))
(dolist (pkg my-packages)
  (unless (package-installed-p pkg)
    (package-refresh-contents)
    (package-install pkg)))

;; ====================== 主题 & 界面优化 =====================
;; 启用语法高亮与行号
(global-font-lock-mode t)
(setq font-lock-maximum-decoration t)
(global-display-line-numbers-mode t)

;; 取消启动画面、隐藏工具栏和菜单栏
(setq inhibit-startup-message t)
(tool-bar-mode -1)
(menu-bar-mode -1)
(set-frame-parameter nil 'internal-border-width 0)
(set-frame-parameter nil 'border-width 0)
(setq ns-use-native-fullscreen t)

;; ====================== 基本快捷键 =====================
(global-set-key [f3] 'revert-buffer)
(global-set-key [f4] 'eval-buffer)
(global-set-key [f5] 'compile)

;; F1 打开 init.el
(global-set-key (kbd "<f1>")
                (lambda ()
                  (interactive)
                  (find-file user-init-file)))

;; Alt 作为 Meta
(setq x-alt-keys-as-meta t)

;; 如果没有打开文件则进入当前目录 dired
(when (and (not (buffer-file-name))
           (eq major-mode 'fundamental-mode))
  (dired "."))

;; ====================== 自定义检查命令 =====================
(defun check-my-config ()
  (interactive)
  (message "【配置状态】")
  (message "当前主题: %s" (car custom-enabled-themes)))
(global-set-key (kbd "C-c C-v") 'check-my-config)

;; ===================== 语言 & 模式 支持 =====================

;; ---- Markdown 支持 ----
(add-to-list 'auto-mode-alist '("\\.md\\'" . markdown-mode))
(with-eval-after-load 'markdown-mode
  (setq markdown-fontify-code-blocks-natively t)
  (add-to-list 'markdown-code-lang-modes '("python" . python-mode)))

;; ---- Python 支持 ----
(add-to-list 'auto-mode-alist '("\\.py\\'" . python-mode))

;; ---- C/C++ 支持 ----
(add-to-list 'auto-mode-alist '("\\.c\\'"   . c-mode))
(add-to-list 'auto-mode-alist '("\\.cpp\\'" . c++-mode))
(add-to-list 'auto-mode-alist '("\\.cc\\'"  . c++-mode))
(add-to-list 'auto-mode-alist '("\\.cxx\\'" . c++-mode))
(setq c-default-style "linux"
      c-basic-offset 4)

;; ---- TypeScript 支持 ----
(add-to-list 'auto-mode-alist '("\\.ts\\'" . typescript-mode))
(unless (package-installed-p 'typescript-mode)
  (package-refresh-contents)
  (package-install 'typescript-mode))
(unless (package-installed-p 'lsp-mode)
  (package-refresh-contents)
  (package-install 'lsp-mode))
(unless (package-installed-p 'lsp-ui)
  (package-refresh-contents)
  (package-install 'lsp-ui))
(unless (package-installed-p 'flycheck)
  (package-refresh-contents)
  (package-install 'flycheck))
(add-hook 'typescript-mode-hook #'lsp-deferred)
(add-hook 'lsp-mode-hook 'flycheck-mode)
(add-hook 'lsp-mode-hook 'lsp-ui-mode)
(add-hook 'typescript-mode-hook
          (lambda ()
            (setq indent-tabs-mode nil
                  tab-width 2)
            (company-mode)))

;; ---- Verilog 支持 ----
(add-to-list 'auto-mode-alist '("\\.sv\\'" . verilog-mode))
(setq verilog-indent-level             2
      verilog-indent-level-module      2
      verilog-indent-level-declaration 2
      verilog-indent-level-behavioral  2
      verilog-cexp-indent              2
      verilog-case-indent              2)
(setq verilog-offset-alist
      '((arglist-cont . 2)
        (arglist-close . 0)))

;; ===================== 自动补全 =====================
(add-hook 'after-init-hook 'global-company-mode)
(add-hook 'python-mode-hook 'company-mode)
(add-hook 'c-mode-hook 'company-mode)
(add-hook 'c++-mode-hook 'company-mode)

;; ===================== Magit / Flycheck 等快捷键 =====================
(global-set-key (kbd "C-c d") 'magit-diff-buffer-file)
(global-set-key (kbd "C-c u") 'uncomment-region)
(global-set-key (kbd "C-c b") 'ibuffer)

;; Swap C-a 和 M-m
(global-set-key (kbd "C-a") #'back-to-indentation)
(global-set-key (kbd "M-m") #'move-beginning-of-line)

;; isearch 区域搜索
(defun my/isearch-forward-region-or-prompt ()
  "如果有选区则搜索选区，否则普通 isearch-forward。"
  (interactive)
  (if (use-region-p)
      (let ((sel (buffer-substring-no-properties
                  (region-beginning) (region-end))))
        (deactivate-mark)
        (isearch-mode t nil nil nil)
        (isearch-yank-string sel))
    (call-interactively #'isearch-forward)))
(global-set-key (kbd "C-s") #'my/isearch-forward-region-or-prompt)

(defun my/isearch-backward-region-or-prompt ()
  "如果有选区则反向搜索选区，否则普通 isearch-backward。"
  (interactive)
  (if (use-region-p)
      (let ((sel (buffer-substring-no-properties
                  (region-beginning) (region-end))))
        (deactivate-mark)
        (isearch-mode nil nil nil nil)
        (isearch-yank-string sel))
    (call-interactively #'isearch-backward)))
(global-set-key (kbd "C-r") #'my/isearch-backward-region-or-prompt)

;; ===================== 窗口大小调整 =====================
(global-set-key (kbd "C-c <down>")  (lambda () (interactive) (shrink-window 2)))
(global-set-key (kbd "C-c <up>")    (lambda () (interactive) (enlarge-window 2)))
(global-set-key (kbd "C-c <right>") (lambda () (interactive) (shrink-window-horizontally 2)))
(global-set-key (kbd "C-c <left>")  (lambda () (interactive) (enlarge-window-horizontally 2)))
(global-set-key (kbd "C-c r")       'query-replace)

;; ===================== 终端 & ansi-term 快捷键 =====================
(with-eval-after-load 'magit
  (define-key magit-diff-mode-map (kbd "RET") #'magit-diff-visit-file-other-window))

(add-hook 'term-mode-hook
          (lambda ()
            (define-key term-raw-map (kbd "C-c b") #'ibuffer)))

(defun my/ansi-term-here ()
  "在当前目录下打开 ansi-term。"
  (interactive)
  (let* ((default-directory
          (or (and (buffer-file-name)
                   (file-name-directory (buffer-file-name)))
              default-directory))
         (shell (or (getenv "SHELL") "/bin/bash")))
    (ansi-term shell)))
(global-set-key (kbd "C-c a") #'my/ansi-term-here)

;; Swap C-x s 与 C-x C-s
(global-set-key (kbd "C-x s")   #'save-buffer)
(global-set-key (kbd "C-x C-s") #'save-some-buffers)

;; delete-selection-mode：选中后输入覆盖
(delete-selection-mode 1)

;; 移除 mark-whole-buffer 的 advice
(advice-remove 'mark-whole-buffer #'my/mwb-keep-point)
(global-set-key (kbd "C-x h") #'mark-whole-buffer)

;; 取消挂起命令
(global-unset-key (kbd "C-z"))
(global-unset-key (kbd "C-x C-z"))

;; 撤销/重做
(global-set-key (kbd "C-z")   #'undo)
(global-set-key (kbd "C-S-z") #'undo-redo)

;; RET 做 newline-and-indent
(global-set-key (kbd "RET") #'newline-and-indent)

;; ===================== 禁用 lsp-mode 和 lsp-ui-mode =====================
(with-eval-after-load 'lsp-mode
  (when (fboundp 'global-lsp-mode) (global-lsp-mode -1))
  (dolist (hook '(javascript-mode-hook
                  typescript-mode-hook
                  python-mode-hook
                  c-mode-hook
                  c++-mode-hook
                  prog-mode-hook))
    (remove-hook hook #'lsp-deferred)
    (remove-hook hook #'lsp))
  (setq lsp-mode-hook nil))
(with-eval-after-load 'lsp-ui
  (setq lsp-ui-doc-enable nil
        lsp-ui-sideline-enable nil
        lsp-ui-imenu-enable nil)
  (add-hook 'lsp-ui-mode-hook (lambda () (lsp-ui-mode -1))))

;; ===================== 缩进优化：Tab 与 Shift+Tab =====================
(transient-mark-mode 1)
(when (and (not (display-graphic-p))
           (string-match "xterm" (getenv "TERM")))
  (define-key input-decode-map "\e[Z" [backtab]))
(global-set-key (kbd "TAB")       #'indent-for-tab-command)
(global-set-key (kbd "<backtab>") #'indent-for-tab-command)
(global-set-key (kbd "<S-Tab>")   #'indent-for-tab-command)
(with-eval-after-load 'python
  (define-key python-mode-map     (kbd "TAB")       #'indent-for-tab-command)
  (define-key python-mode-map     (kbd "<backtab>") #'indent-for-tab-command)
  (define-key python-mode-map     (kbd "<S-Tab>")   #'indent-for-tab-command))
(with-eval-after-load 'python-ts-mode
  (define-key python-ts-mode-map  (kbd "TAB")       #'indent-for-tab-command)
  (define-key python-ts-mode-map  (kbd "<backtab>") #'indent-for-tab-command)
  (define-key python-ts-mode-map  (kbd "<S-Tab>")   #'indent-for-tab-command))

;; ------------------------------------------------------------
; Tell Emacs to use ripgrep instead of its default grep command
; M-x grep-find
; see https://stegosaurusdormant.com/emacs-ripgrep/
;; ------------------------------------------------------------
(setq grep-find-command
      '("rg -n -H --no-heading -e '' $(git rev-parse --show-toplevel || pwd)" . 27))
;; make C-v do the same as C-y (paste)
(global-set-key (kbd "C-v") #'yank)

;; ===================== End of init.el =====================
