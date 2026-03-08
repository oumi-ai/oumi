(function() {
    'use strict';

    document.addEventListener('DOMContentLoaded', function() {
        initSmoothScrolling();
        initCodeBlockEnhancements();
        initTableOfContentsHighlight();
        initSearchEnhancements();
        initKeyboardNavigation();
    });

    function initSmoothScrolling() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                if (href === '#') return;

                const target = document.querySelector(href);
                if (target) {
                    e.preventDefault();
                    const headerOffset = 80;
                    const elementPosition = target.getBoundingClientRect().top;
                    const offsetPosition = elementPosition + window.scrollY - headerOffset;

                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });

                    history.pushState(null, null, href);
                }
            });
        });
    }

    function initCodeBlockEnhancements() {
        document.querySelectorAll('div[class*="highlight-"]').forEach(block => {
            const classes = block.className.split(' ');
            const langClass = classes.find(c => c.startsWith('highlight-'));
            if (langClass) {
                const lang = langClass.replace('highlight-', '').toUpperCase();
                if (lang && lang !== 'DEFAULT') {
                    const label = document.createElement('span');
                    label.className = 'oumi-code-lang';
                    label.textContent = lang;
                    label.style.cssText = `
                        position: absolute;
                        top: 8px;
                        right: 8px;
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 0.65rem;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        color: var(--oumi-text-muted, #6E7681);
                        background: var(--oumi-bg-elevated, #242B35);
                        padding: 4px 8px;
                        border-radius: 4px;
                        opacity: 0.8;
                        pointer-events: none;
                    `;
                    block.style.position = 'relative';
                    block.appendChild(label);
                }
            }
        });

        document.querySelectorAll('button.copybtn').forEach(btn => {
            btn.addEventListener('click', function() {
                const originalBg = this.style.backgroundColor;
                const originalBorder = this.style.borderColor;

                this.style.backgroundColor = 'var(--oumi-success, #3FB950)';
                this.style.borderColor = 'var(--oumi-success, #3FB950)';

                setTimeout(() => {
                    this.style.backgroundColor = originalBg;
                    this.style.borderColor = originalBorder;
                }, 2000);
            });
        });
    }

    function initTableOfContentsHighlight() {
        const tocLinks = document.querySelectorAll('.bd-toc .nav-link');
        if (tocLinks.length === 0) return;

        const sections = [];
        tocLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                const section = document.querySelector(href);
                if (section) {
                    sections.push({ link, section });
                }
            }
        });

        function highlightActiveSection() {
            const scrollPos = window.scrollY + 100;

            let activeSection = null;
            sections.forEach(({ link, section }) => {
                if (section.offsetTop <= scrollPos) {
                    activeSection = link;
                }
            });

            tocLinks.forEach(link => {
                link.classList.remove('toc-active');
                link.style.color = '';
                link.style.fontWeight = '';
            });

            if (activeSection) {
                activeSection.classList.add('toc-active');
                activeSection.style.color = 'var(--oumi-accent, #00D4AA)';
                activeSection.style.fontWeight = '600';
            }
        }

        window.addEventListener('scroll', highlightActiveSection, { passive: true });
        highlightActiveSection();
    }

    function initSearchEnhancements() {
        const searchInput = document.querySelector('.bd-search input, input[type="search"]');
        if (searchInput) {
            const hint = document.createElement('kbd');
            hint.textContent = '/';
            hint.style.cssText = `
                position: absolute;
                right: 12px;
                top: 50%;
                transform: translateY(-50%);
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                padding: 2px 6px;
                background: var(--oumi-bg-elevated, #242B35);
                border: 1px solid var(--oumi-border, #30363D);
                border-radius: 4px;
                color: var(--oumi-text-muted, #6E7681);
                pointer-events: none;
            `;

            const searchContainer = searchInput.closest('.bd-search');
            if (searchContainer) {
                searchContainer.style.position = 'relative';
                searchContainer.appendChild(hint);

                searchInput.addEventListener('focus', () => hint.style.display = 'none');
                searchInput.addEventListener('blur', () => {
                    if (!searchInput.value) hint.style.display = '';
                });
            }
        }
    }

    function initKeyboardNavigation() {
        document.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                if (e.key === 'Escape') {
                    e.target.blur();
                }
                return;
            }

            if (e.key === '/') {
                const searchInput = document.querySelector('.bd-search input, input[type="search"]');
                if (searchInput) {
                    e.preventDefault();
                    searchInput.focus();
                }
            }

            if (e.key === 't' && !e.ctrlKey && !e.metaKey) {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }

            if (e.key === 'g') {
                const waitForH = function(e2) {
                    if (e2.key === 'h') {
                        window.location.href = '/';
                    }
                    document.removeEventListener('keydown', waitForH);
                };
                setTimeout(() => document.removeEventListener('keydown', waitForH), 1000);
                document.addEventListener('keydown', waitForH);
            }
        });
    }

    function initThemeTransition() {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.attributeName === 'data-theme') {
                    document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
                    setTimeout(() => {
                        document.body.style.transition = '';
                    }, 300);
                }
            });
        });

        observer.observe(document.documentElement, { attributes: true });
    }

    initThemeTransition();

})();
