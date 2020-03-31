% ==============================================================================
% Identification
% ==============================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{shtthesis}[%
  2020/03/28 %
  0.2.0 %
  An Unofficial Thesis Template for ShanghaiTech University%
]
\hyphenation{sht-thesis}
\hyphenation{Sht-Thesis}
\RequirePackage{xspace}
\newcommand\shtthesis{\textup{\sffamily shtthesis}\xspace}
\newcommand\ShtThesis{\textup{\sffamily ShtThesis}\xspace}
\newcommand\version{0.2.0\xspace}
% ==============================================================================
% Options and Declarations
% ==============================================================================
\RequirePackage{kvdefinekeys}
\RequirePackage{kvsetkeys}
\RequirePackage{kvoptions}
\RequirePackage{datetime}
% ------------------------------------------------------------------------------
%   Define \shtsetup command
%
%   Following key-value setup mechanism is adapted from `thuthesis'.
%   thuthesis (https://github.com/xueruini/thuthesis) is licensed under the
%   conditions of the LaTeX Project Public License, either version 1.3. 
% ------------------------------------------------------------------------------
\SetupKeyvalOptions{
  family = sht,
  prefix = {sht@},
  setkeys = \kvsetkeys,
}
\newcommand\sht@error[1]{%
  \ClassError{shtthesis}{#1}{}%
}
\newcommand\shtsetup{%
  \kvsetkeys{sht}%
}
\newcommand\sht@define@key[1]{%
  \kvsetkeys{sht@key}{#1}%
}
\kv@set@family@handler{sht@key}{%
  \@namedef{sht@#1@@name}{#1}%
  \def\sht@@default{}%
  \def\sht@@choices{}%
  \kv@define@key{sht@value}{name}{%
    \@namedef{sht@#1@@name}{##1}%
  }%
  \kv@define@key{sht@value}{code}{%
    \@namedef{sht@#1@@code}{##1}%
  }%
  \@namedef{sht@#1@@check}{}%
  \@namedef{sht@#1@@code}{}%
  \@namedef{sht@#1@@hook}{%
    \expandafter\ifx\csname\@currname.\@currext-h@@k\endcsname\relax
      \@nameuse{sht@#1@@code}%
    \else
      \AtEndOfClass{%
        \@nameuse{sht@#1@@code}%
      }%
    \fi
  }%
  \kv@define@key{sht@value}{choices}{%
    \def\sht@@choices{##1}%
    \@namedef{sht@#1@@reset}{}%
    \@namedef{sht@#1@@check}{%
      \@ifundefined{%
        ifsht@\@nameuse{sht@#1@@name}@\@nameuse{sht@\@nameuse{sht@#1@@name}}%
      }{%
        \sht@error{Invalid value "#1 = \@nameuse{sht@\@nameuse{sht@#1@@name}}"}%
      }%
      \@nameuse{sht@#1@@reset}%
      \@nameuse{sht@\@nameuse{sht@#1@@name}@\@nameuse{sht@\@nameuse{sht@#1@@name}}true}%
    }%
  }%
  \kv@define@key{sht@value}{default}{%
    \def\sht@@default{##1}%
  }%
  \kvsetkeys{sht@value}{#2}%
  \@namedef{sht@\@nameuse{sht@#1@@name}}{}%
  \kv@set@family@handler{sht@choice}{%
    \ifx\sht@@default\@empty
      \def\sht@@default{##1}%
    \fi
    \expandafter\newif\csname ifsht@\@nameuse{sht@#1@@name}@##1\endcsname
    \expandafter\g@addto@macro\csname sht@#1@@reset\endcsname{%
      \@nameuse{sht@\@nameuse{sht@#1@@name}@##1false}%
    }%
  }%
  \kvsetkeys@expandafter{sht@choice}{\sht@@choices}%
  \expandafter\let\csname sht@\@nameuse{sht@#1@@name}\endcsname\sht@@default
  \expandafter\ifx\csname sht@\@nameuse{sht@#1@@name}\endcsname\@empty\else
    \@nameuse{sht@#1@@check}%
    \@nameuse{sht@#1@@hook}%
  \fi
  \kv@define@key{sht}{#1}{%
    \@namedef{sht@\@nameuse{sht@#1@@name}}{##1}%
    \@nameuse{sht@#1@@check}%
    \@nameuse{sht@#1@@hook}%
  }%
}
% ------------------------------------------------------------------------------
%   Options for \shtsetup
% ------------------------------------------------------------------------------
\sht@define@key{
  degree = {
    choices = {
      bachelor,
      master,
      doctor,
      postdoc,
    },
    default = doctor,
  },
  degree-type = {
    choices = {
      academic,
      professional,
    },
    name = degree@type,
  },
  degree-name = {
    default = {工学博士},
    name    = degree@name,
  },
  degree-name* = {
    default = {Doctor~of~Philosophy},
    name    = degree@name@en,
  },
  language = {
    choices = {
      chinese,
      english,
    },
  },
  secret-level = {
    name = secret@level,
  },
  secret-year = {
    name = secret@year,
  },
  title = {
    default = {标题},
  },
  title* = {
    default = {Title},
    name    = title@en,
  },
  keywords,
  keywords* = {
    name = keywords@en,
  },
  author = {
    default = {姓名},
  },
  author* = {
    default = {Name of author},
    name    = author@en,
  },
  supervisor = {
    default = {导师姓名},
  },
  supervisor* = {
    default = {Name of supervisor},
    name    = supervisor@en,
  },
  supervisor-institution = {
    name = supervisor@institution,
  },
  supervisor-institution* = {
    name = supervisor@institution@en,
  },
  associate-supervisor = {
    name = associate@supervisor,
  },
  associate-supervisor* = {
    name = associate@supervisor@en,
  },
  associate-supervisor-institution = {
    name = associate@supervisor@institution,
  },
  associate-supervisor-institution* = {
    name = associate@supervisor@institution@en,
  },
  joint-supervisor = {
    name = joint@supervisor,
  },
  joint-supervisor* = {
    name = joint@supervisor@en,
  },
  joint-supervisor-institution = {
    name = joint@supervisor@institution,
  },
  joint-supervisor-institution* = {
    name = joint@supervisor@institution@en,
  },
  institution = {
    % default = {上海科技大学信息科学与技术学院},
  },
  institution* = {
    % default = {%
    %   School~of~Information~Science~and~Technology\\%
    %   ShanghaiTech~University%
    % },
    name = institution@en
  },
  discipline = {
    % default = {计算机科学与技术},
  },
  discipline* = {
    % default = {Computer Science and Technology},
    name = discipline@en,
  },
  date = {
    default = {\the\year~年~\the\month~月},
  },
  date* = {
    default = {\monthname,~\the\year},
    name = date@en,
  },
  clc,
  udc,
  id,
  discipline-level-1 = {
    default = {一级学科名称},
    name    = discipline@level@i,
  },
  discipline-level-1* = {
    default = {Name of Level-one Discipline},
    name    = discipline@level@i@en,
  },
  discipline-level-2 = {
    default = {二级学科名称},
    name    = discipline@level@ii,
  },
  discipline-level-2* = {
    default = {Name of Level-two Discipline},
    name    = discipline@level@ii@en,
  },
  start-date = {
    name    = start@date,
    default = {\the\year-\two@digits{\month}-\two@digits{\day}},
  },
  end-date = {
    name    = end@date,
    default = {\the\year-\two@digits{\month}-\two@digits{\day}},
  },
  anonymous-str = {
    name = anonymous@str,
    default = {***},
  }
}
% ------------------------------------------------------------------------------
%   Class options
% ------------------------------------------------------------------------------
\SetupKeyvalOptions{
  family = sht,
  prefix = sht@,
}
\DeclareBoolOption[false]{anonymous}
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{ctexbook}}
\ProcessKeyvalOptions*
% ------------------------------------------------------------------------------
%   Load minimum mandatory packages
% ------------------------------------------------------------------------------
\LoadClass[
  UTF8,
  a4paper,
  twoside, 
  openany, 
  zihao = -4, 
  scheme = plain, 
]{ctexbook}
\RequirePackage{geometry}
\RequirePackage{calc}
\RequirePackage{verbatim}
\RequirePackage{etoolbox}
\RequirePackage{ifthen}
\RequirePackage{graphicx}
\RequirePackage{indentfirst}
\RequirePackage[normalem]{ulem}
\RequirePackage{fancyhdr}
\RequirePackage{tocvsec2}
\RequirePackage{letltxmacro}
\RequirePackage{fontspec}[2016/12/27]
\RequirePackage{caption}
\RequirePackage{natbib}
\RequirePackage[shortlabels, inline]{enumitem}
\RequirePackage{amsmath}
\RequirePackage{mathtools}
\RequirePackage[
  mathrm = sym,
  mathit = sym,
  mathbf = sym,
  mathtt = sym,
]{unicode-math}
\AtEndPreamble{
  \RequirePackage{hyperref}
  \hypersetup{
    pdfencoding = auto,
    psdextra = true,
    bookmarksnumbered = true,
    pdftitle = {\sht@title},
    pdfauthor = {\sht@author},
    colorlinks = true,
    linkcolor = black,
    citecolor = blue,
    urlcolor = blue,
  }
}
% ==============================================================================
% Format Specifications
% ==============================================================================
% ------------------------------------------------------------------------------
%   Basic information
% ------------------------------------------------------------------------------
\def\@title{\sht@title}
\def\@author{\sht@author}
\def\contentsname{目\hspace{1\ccwd}录}
\def\listfigurename{图形列表}
\def\listtablename{表格列表}
\def\appendixname{附录}
\def\indexname{索引}
\def\refname{参考文献}
\def\bibname{参考文献}
\def\bibetal{等}
\def\biband{和}
\def\tablename{表}
\def\figurename{图}
% ------------------------------------------------------------------------------
%   Geometry and layouts
% ------------------------------------------------------------------------------
\geometry{
  inner = 3.17cm,
  outer = 3.17cm,
  top = 2.54cm,
  bottom = 2.54cm,
  headheight = 12pt, % 9bp,
  headsep = 17.5pt, % 2.54cm - 9bp,
  footskip = 2.54cm - 1.75cm,
}
\setlength{\parskip}{0.5ex plus 0.25ex minus 0.25ex}
\setlength{\parindent}{2\ccwd}
\linespread{1.6}
\raggedbottom
% ------------------------------------------------------------------------------
%   Headers and footers
% ------------------------------------------------------------------------------
\pagestyle{fancy}
\fancypagestyle{CenteralPageNumbered}{
  \fancyhf{}
  \pagenumbering{Roman}
  \fancyhead[C]{
    \footnotesize
    \ifnumodd{\value{page}}{
      \nouppercase\leftmark
    }{
      \nouppercase\@title
    }
  }
  \fancyfoot[C]{
    \footnotesize
    \thepage
  }
  \renewcommand{\headrulewidth}{0.8pt}
  \renewcommand{\footrulewidth}{0pt}
}
\fancypagestyle{LRPageNumbered}{
  \fancyhf{}
  \fancyhead[C]{
    \footnotesize
    \ifnumodd{\value{page}}{
      \nouppercase\leftmark
    }{
      \nouppercase\@title
    }
  }
  \fancyfoot[C]{
    \footnotesize
    \ifnumodd{\value{page}}{
      \hspace*{\fill}
      \thepage
    }{
      \thepage
      \hspace*{\fill}
    }
  }
  \renewcommand{\headrulewidth}{0.8pt}
  \renewcommand{\footrulewidth}{0pt}
}
\fancypagestyle{LRPageNumberedAppendix}{
  \fancyhf{}
  \fancyhead[C]{
    \footnotesize
    \ifnumodd{\value{page}}{%
      附\hspace{1\ccwd}录%
    }{
      \nouppercase\@title
    }
  }
  \fancyfoot[C]{
    \footnotesize
    \ifnumodd{\value{page}}{
      \hspace*{\fill}
      \thepage
    }{
      \thepage
      \hspace*{\fill}
    }
  }
  \renewcommand{\headrulewidth}{0.8pt}
  \renewcommand{\footrulewidth}{0pt}
}
\fancypagestyle{LRPageNumberedWithoutHead}{
  \fancyhf{}
  \fancyfoot[C]{
    \footnotesize
    \ifnumodd{\value{page}}{
      \hspace*{\fill}
      \thepage
    }{
      \thepage
      \hspace*{\fill}
    }
  }
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}
\fancypagestyle{Plain}{}
\providecommand{\frontmatter}{}
\LetLtxMacro{\TmpFrontmatter}{\frontmatter}
\renewcommand{\frontmatter}{%
  \TmpFrontmatter%
  \pagestyle{CenteralPageNumbered}%
}
\providecommand{\mainmatter}{}
\LetLtxMacro{\TmpMainmatter}{\mainmatter}
\renewcommand{\mainmatter}{%
  \TmpMainmatter%
  \pagestyle{LRPageNumbered}%
}
% ------------------------------------------------------------------------------
%   Font utilities
% ------------------------------------------------------------------------------
\newcommand\sht@fontset{\csname g__ctex_fontset_tl\endcsname}
\newcommand\currentfontset{\sht@fontset\xspace}
% ------------------------------------------------------------------------------
%   English fonts
% ------------------------------------------------------------------------------
\setmainfont{texgyretermes} [
  Extension      = .otf,
  UprightFont    = *-regular,
  BoldFont       = *-bold,
  ItalicFont     = *-italic,
  BoldItalicFont = *-bolditalic,
  Ligatures      = TeX,
]
\setsansfont{texgyreheros} [
  Extension      = .otf,
  UprightFont    = *-regular,
  BoldFont       = *-bold,
  ItalicFont     = *-italic,
  BoldItalicFont = *-bolditalic,
  Ligatures      = TeX,
]
\setmonofont{texgyrecursor} [
  Extension      = .otf,
  UprightFont    = *-regular,
  BoldFont       = *-bold,
  ItalicFont     = *-italic,
  BoldItalicFont = *-bolditalic,
  Ligatures      = TeX,
]
% ------------------------------------------------------------------------------
%   Chinese fonts
% ------------------------------------------------------------------------------
\ifthenelse{\equal{\sht@fontset}{windows}}{
  \def\fake@bold@factor{2.5}
  \IfFileExists{C:/bootfont.bin}{
    \def\windows@kaiti{KaiTi_GB2312}
  }{
    \def\windows@kaiti{KaiTi}
  }
  \setCJKmainfont[
      AutoFakeBold = \fake@bold@factor,
        ItalicFont = \windows@kaiti,
    ]{SimSun}
  \setCJKsansfont[
      AutoFakeBold = \fake@bold@factor,
    ]{SimHei}
  \setCJKfamilyfont{zhsong}[
      AutoFakeBold = \fake@bold@factor,
    ]{SimSun}
  \setCJKfamilyfont{zhhei}[
      AutoFakeBold = \fake@bold@factor,
    ]{SimHei}
  \setCJKfamilyfont{zhkai}[
      AutoFakeBold = \fake@bold@factor,
    ]{\windows@kaiti}
}{
  \ifthenelse{\equal{\sht@fontset}{mac}}{
    \setCJKmainfont[
           UprightFont = * Light,
              BoldFont = * Bold,
            ItalicFont = Kaiti SC Regular,
        BoldItalicFont = Kaiti SC Bold,
      ]{Songti SC}
    \setCJKsansfont[
        UprightFont = * Light,
           BoldFont = * Medium,
      ]{Heiti SC}
    \setCJKfamilyfont{zhsong}[
        UprightFont = * Light,
           BoldFont = * Bold,
      ]{Songti SC}
    \setCJKfamilyfont{zhhei}[
        UprightFont = * Light,
           BoldFont = * Medium,
      ]{Heiti SC}
    \setCJKfamilyfont{zhkai}[
        UprightFont = * Regular,
           BoldFont = * Bold,
      ]{Kaiti SC}
  }{}
}
% ------------------------------------------------------------------------------
%   Math content specifications
%
%   Following setup is adapted from `thuthesis'
% ------------------------------------------------------------------------------
\unimathsetup{
  math-style = ISO,
  bold-style = ISO,
  nabla      = upright,
  partial    = upright,
}
\protected\def\le{\leqslant}
\protected\def\ge{\geqslant}
\AtBeginDocument{%
  \renewcommand\leq{\leqslant}%
  \renewcommand\geq{\geqslant}%
}
\removenolimits{%
  \int\iint\iiint\iiiint\oint\oiint\oiiint
  \intclockwise\varointclockwise\ointctrclockwise\sumint
  \intbar\intBar\fint\cirfnint\awint\rppolint
  \scpolint\npolint\pointint\sqint\intlarhk\intx
  \intcap\intcup\upint\lowint
}
\AtBeginDocument{%
  \renewcommand{\Re}{\operatorname{Re}}%
  \renewcommand{\Im}{\operatorname{Im}}%
}
\newcommand\bm{\symbf}
\renewcommand\boldsymbol{\symbf}
\newcommand\square{\mdlgwhtsquare}
\AtBeginDocument{%
  \renewcommand\checkmark{\ensuremath{✓}}%
}
\allowdisplaybreaks[4]
\newtagform{dots}{\ldots\ (}{)}
\usetagform{dots}
\renewcommand{\eqref}[1]{\textup{(\ref{#1})}}
% ------------------------------------------------------------------------------
%   Math fonts
% ------------------------------------------------------------------------------
\IfFontExistsTF{XITSMath-Regular.otf}{
  \setmathfont{XITSMath-Regular} [
    Extension    = .otf,
    BoldFont     = XITSMath-Bold,
    StylisticSet = 8,
  ]
  \setmathfont{XITSMath-Regular.otf} [
    range        = {cal, bfcal},
    StylisticSet = 1,
  ]
  \setmathfont{XITSMath-Regular.otf} [
    range        = {scr, bfscr},
    StylisticSet = 2,
  ]
}{
  \setmathfont{xits-math} [
    Extension    = .otf,
    BoldFont     = *bold,
    StylisticSet = 8,
  ]
  \setmathfont{xits-math.otf} [
    range        = {cal, bfcal},
    StylisticSet = 1,
  ]
  \setmathfont{xits-math.otf} [
    range        = {scr, bfscr},
    StylisticSet = 2,
  ]
}
\IfFontExistsTF{FiraMath-Regular.otf}{
  \setmathfontface\mathsf{FiraMath-Regular.otf}
}{}
% ------------------------------------------------------------------------------
%   Utility functions for ToC and bookmarks
% ------------------------------------------------------------------------------
\newcommand{\artxmaincnt}{%
  \ifcsname c@chapter\endcsname%
    chapter%
  \else%
    section%
  \fi%
}
\long\def\artxaux#1{}
\newcommand{\intotocnostar}[3]{%
  % [<\chapter|...>,<*>,<title>]
  #1#2{%
    \phantomsection%
    \addcontentsline{toc}{\expandafter\artxaux\string#1}{#3}%
    #3%
  }
  \markboth{\MakeUppercase{#3}}{}%
}
\newcommand{\intotocstar}[3][\artxmaincnt]{%
  % [<chapter|...>,<extra commands>,<title>]
  #2%
  \phantomsection%
  \addcontentsline{toc}{#1}{#3}%
  \markboth{\MakeUppercase{#3}}{}%
}
\newcommand{\intotoc}{\@ifstar{\intotocstar}{\intotocnostar}}
\newcommand{\intobmknostar}[4][0]{%
  % [<level>,<\chapter|...>,<*>,<title>]
  #2#3{%
    \phantomsection%
    \Hy@writebookmark%
    {}%
    {#4}%
    {\@currentHref}%
    {#1}%
    {toc}%
    #4%
  }%
  \markboth{\MakeUppercase{#4}}{}%
}
\newcommand{\intobmkstar}[3][0]{%
  % [<level>,<extra commands>,<title>]
  #2%
  \phantomsection%
  \Hy@writebookmark%
  {}%
  {#3}%
  {\@currentHref}%
  {#1}%
  {toc}%
  \markboth{\MakeUppercase{#3}}{}%
}
\newcommand{\intobmk}{\@ifstar{\intobmkstar}{\intobmknostar}}
% ------------------------------------------------------------------------------
%   Table of contents, figures and tables
% ------------------------------------------------------------------------------
\DeclareCaptionFont{wuhaocuti}{\zihao{5}\bfseries}
\captionsetup{
  format = plain,
  hangindent = 2.0em,
  skip = 8pt,
  labelsep = quad,
  font = {wuhaocuti},
}
\def\@dotsep{1.5mu}
\def\@pnumwidth{2em}
\def\@tocrmarg{2em}
\def\@chaptervspace{1ex}
% [<level>,<indent>,<numwidth>,<title>,<page>]
\renewcommand*{\@dottedtocline}[5]{
  \ifnum #1>\c@tocdepth \else
    \vskip \z@ \@plus.2\p@
    {\leftskip #2\relax \rightskip \@tocrmarg \parfillskip -\rightskip
    \parindent #2\relax\@afterindenttrue
    \interlinepenalty\@M
    \leavevmode \zihao{-4}\heiti
    \@tempdima #3\relax
    \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
    {#4}\nobreak
    \leaders\hbox{$\m@th\mkern \@dotsep \cdot\mkern \@dotsep$}\hfill
    \nobreak
    \hb@xt@\@pnumwidth{\hfil\normalfont \normalcolor #5}%
    \par\penalty\@highpenalty}%
  \fi
}
% [<title>,<page>]
\renewcommand*{\l@part}[2]{
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{3em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
      \zihao{4}\heiti #1
      \leaders\hbox{$\m@th\mkern \@dotsep \cdot\mkern \@dotsep$}
      \hfil \hb@xt@\@pnumwidth{\hss #2}}\par
      \nobreak
      \global\@nobreaktrue
      \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi
}
% [<title>,<page>]
\renewcommand*{\l@chapter}[2]{
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip \@chaptervspace \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \zihao{4}\heiti
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak
      \leaders\hbox{$\m@th\mkern \@dotsep \cdot\mkern \@dotsep$}
      \hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi
}
\renewcommand*\l@section{\@dottedtocline{1}{1em}{1.8em}}
\renewcommand*\l@subsection{\@dottedtocline{2}{2em}{2.8em}}
\renewcommand*\l@subsubsection{\@dottedtocline{3}{3em}{3.8em}}
\renewcommand*\l@paragraph{\@dottedtocline{4}{4em}{4.8em}}
\renewcommand*\l@subparagraph{\@dottedtocline{5}{5em}{5.8em}}
\renewcommand*\l@figure{\@dottedtocline{1}{1em}{1.8em}}
\renewcommand*\l@table{\@dottedtocline{1}{1em}{1.8em}}
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{3}
\newcommand{\makeindices}{%
  \begingroup%
  \linespread{1.2}%
  \intobmk*{\cleardoublepage}{\contentsname}%
  \tableofcontents%
  \intobmk*{\cleardoublepage}{\listfigurename}%
  \listoffigures%
  \intobmk*{\cleardoublepage}{\listtablename}%
  \listoftables%
  \endgroup%
}
% ------------------------------------------------------------------------------
%   Content specifications
% ------------------------------------------------------------------------------
\ctexset {
  chapter = {
    format = \bfseries\heiti\zihao{4}\linespread{1.0}\centering,
    nameformat = {},
    numberformat = \rmfamily,
    titleformat = {},
    name = {第, 章},
    number = \arabic{chapter},
    aftername = \hspace{1\ccwd},
    beforeskip = {7pt},
    afterskip = {18pt},
    pagestyle = Plain,
  },
  section = {
    format = \heiti\normalsize\linespread{1.0}\raggedright,
    nameformat = {},
    numberformat = \rmfamily,
    titleformat = {},
    aftername = \hspace{1\ccwd},
    beforeskip = {24pt},
    afterskip = {6pt},
  },
  subsection = {
    format = \heiti\normalsize\linespread{1.0}\raggedright,
    nameformat = {},
    numberformat = \rmfamily,
    titleformat = {},
    aftername = \hspace{1\ccwd},
    beforeskip = {12pt},
    afterskip = {6pt},
  },
  paragraph = {
    indent = 2\ccwd,
  },
  punct = quanjiao,
  space = auto,
  autoindent = true,
}
% ------------------------------------------------------------------------------
%   List environments
% ------------------------------------------------------------------------------
\setlist{nosep}
% ==============================================================================
% Cover Generation
% ==============================================================================
\def\ulhshift{-4em}
\newcommand{\ulunify}[1]{\uline{#1}}
\newcommand{\ulenhance}[2][1pt]{%
  \def\ULthickness{#1}%
  \setlength{\ULdepth}{0.3em}%
  \expandafter\ulunify\expandafter{#2}%
}
\newcommand{\ulextend}[2][350pt]{%
  \hbox to #1 {\hfill \hspace*{\ulhshift} #2 \hfill}%
}
\newcommand\sht@schoollogo{%
  \includegraphics[width=10.48cm]{shanghaitech_logo.pdf}%
}
\renewcommand{\maketitle}{%
  \ifthenelse{\equal{\sht@degree}{doctor}}{
    \sht@maketitle@graduate{博士}
    \sht@maketitle@graduate@en{dissertation}
  }{
    \ifthenelse{\equal{\sht@degree}{master}}{
      \sht@maketitle@graduate{硕士}
      \sht@maketitle@graduate@en{thesis}
    }{
      \sht@error{Degree type `\sht@degree' is not supported yet}
    }
  }
  \ifthenelse{\not \boolean{sht@anonymous}}{%
    \makedeclarations%
  }{}
}
% ------------------------------------------------------------------------------
%   PhD/Master covers
% ------------------------------------------------------------------------------
\newcommand{\sht@maketitle@graduate}[1]{%
  \intobmk*{\cleardoublepage}{封面}
  \thispagestyle{empty}
  \begin{center}
    \linespread{1.6}\zihao{4}\bfseries
    \hfill{}
    \ifdefempty{\sht@secret@level}{}{%
      密级：%
      \ulenhance{%
        \ulextend[50pt]{%
          \hspace*{-\ulhshift} \zihao{5} \sht@secret@level%
        }%
      }%
    }

    \vspace*{\stretch{4}}

    \sht@schoollogo

    \vspace*{\stretch{2}}

    {\zihao{1}\heiti #1学位论文}

    \vspace*{\stretch{3}}

    {\zihao{-3}\heiti \ulenhance[1.5pt]{\sht@title}}

    \vspace*{\stretch{3}}

    {
      \songti
      \def\tabcolsep{1pt}
      \def\arraystretch{1.3}
      \begin{tabular}{llc}
        作者姓名：&
          \multicolumn{2}{c}{\ulenhance[1.2pt]{\ulextend{%
            \ifsht@anonymous%
              \sht@anonymous@str%
            \else%
              \sht@author%
            \fi%
          }}} \\
        指导教师：&
          \multicolumn{2}{c}{\ulenhance[1.2pt]{\ulextend{%
            \ifsht@anonymous%
              \sht@anonymous@str%
            \else%
              \sht@supervisor%
            \fi%
          }}} \\
        \ifdefempty{\sht@supervisor@institution}{}{ &
          \multicolumn{2}{c}{\ulenhance[1.2pt]{\ulextend{\sht@supervisor@institution}}} \\
        }
        学位类别：&
          \multicolumn{2}{c}{\ulenhance[1.2pt]{\ulextend{\sht@degree@name}}} \\
        一级学科：&
          \multicolumn{2}{c}{\ulenhance[1.2pt]{\ulextend{\sht@discipline@level@i}}} \\
        \multicolumn{2}{l}{学校/学院名称：} &
          \ulenhance[1.2pt]{\ulextend[318pt]{\sht@institution}} \\
      \end{tabular}
    }

    \vspace*{\stretch{4}}

    \sht@date

    \vspace{\stretch{4}}
  \end{center}
  \clearpage
  \thispagestyle{empty}
  \cleardoublepage
}
\newcommand{\sht@maketitle@graduate@en}[1]{
  \intobmk*{\cleardoublepage}{Cover}
  \thispagestyle{empty}
  \begin{center}
    \linespread{1.6}
    \zihao{4}\bfseries

    \vspace*{50pt}

    {\zihao{-3}\bfseries \ulenhance[1.5pt]{\sht@title@en}}

    \vspace*{\stretch{3}}

    {
      A~#1~submitted~to\\
      ShanghaiTech~University\\
      in~partial~fulfillment~of~the~requirement\\
      for~the~degree~of\\
      \ifthenelse{\equal{\sht@degree}{doctor}}{
        Doctor~of~Philosophy\\
      }{
        \sht@degree@name@en\\
      }
      in~\sht@discipline@level@i@en

      By

      \ifsht@anonymous%
        \sht@anonymous@str%
      \else%
        \sht@author@en%
      \fi%

      Supervisor:~%
      \ifsht@anonymous%
        \sht@anonymous@str%
      \else%
        \sht@supervisor@en%
      \fi%
    }

    \vspace*{\stretch{3}}

    {\sht@institution@en}

    \vspace*{\stretch{1}}

    {\sht@date@en}

    \vspace{\stretch{3}}
  \end{center}
  \clearpage
  \thispagestyle{empty}
  \cleardoublepage
}
% ------------------------------------------------------------------------------
%   Declarations
% ------------------------------------------------------------------------------
\newcommand{\makedeclarations}{%
  \intobmk*{\cleardoublepage}{声明}
  \thispagestyle{empty}
  {
    \linespread{1.6}\zihao{-4}

    \vspace*{2ex}

    \begin{center}
      {\zihao{4}\bfseries\sffamily 上海科技大学\\研究生学位论文原创性声明}
    \end{center}

    本人郑重声明:所呈交的学位论文是本人在导师的指导下独立进行研究工作所取得的成果。%
    尽我所知，除文中已经注明引用的内容外，本论文不包含任何其他个人或集体已经发表或撰写%
    过的研究成果。对论文所涉及的研究工作做出贡献的其他个人和集体，均已在文中以明确方式%
    标明或致谢。

    \vspace*{3ex}

    {\hfill{} {作者签名：\hspace*{14em}}}

    {\hfill{} {日\quad\quad 期：\hspace*{14em}}}

    \vspace*{6ex}

    \begin{center}
      {\zihao{4}\bfseries\sffamily 上海科技大学\\学位论文授权使用声明}
    \end{center}

    本人完全了解并同意遵守上海科技大学有关保存和使用学位论文的规定，即上海科技大学有权%
    保留送交学位论文的副本，允许该论文被查阅，可以按照学术研究公开原则和保护知识产权%
    的原则公布该论文的全部或部分内容，可以采用影印、缩印或其他复制手段保存、汇编本%
    学位论文。
    
    涉密及延迟公开的学位论文在解密或延迟期后适用本声明。

    \vspace*{3ex}

    {\hfill{} {作者签名：\hspace*{10em} 导师签名：\hspace*{9em}}}

    {\hfill{} {日\quad\quad 期：\hspace*{10em} 日\quad\quad 期：\hspace*{9em}}}

    \vspace{3ex}
  }
  \clearpage
  \thispagestyle{empty}
  \cleardoublepage
}
% ------------------------------------------------------------------------------
%   Abstract
% ------------------------------------------------------------------------------
\newenvironment{abstract}{%
  \cleardoublepage%
  \intobmk\chapter*{摘\hspace{1\ccwd}要}%
}{%
  \vspace{\baselineskip}%
  \par\noindent{\bfseries 关键词：} \sht@keywords%
}
\newenvironment{abstract*}{%
  \cleardoublepage%
  \intobmk\chapter*{Abstract}%
}{%
  \vspace{\baselineskip}%
  \par\noindent{\bfseries Key Words:} \sht@keywords@en%
}
% ------------------------------------------------------------------------------
%   Nomenclatures
% ------------------------------------------------------------------------------
\newcounter{sht@nomenclature@cnt}
\setcounter{sht@nomenclature@cnt}{0}
\def\sht@null@arg{}
\newenvironment{nomenclatures}[1][\sht@null@arg]{%
  \ifnum\thesht@nomenclature@cnt=0
    \cleardoublepage
    \intobmk\chapter*{符号列表}%
  \fi
  \stepcounter{sht@nomenclature@cnt}
  \ifthenelse{\not \equal{#1}{\sht@null@arg}}{
    \section*{#1}
  }{
    \par\vspace{24pt}
  }
  \renewcommand{\item}[3][\sht@null@arg]{
    \ifthenelse{\not \equal{##1}{\sht@null@arg}}{
      \par\noindent\makebox[0.15\columnwidth][l]{##2}{{##3}\hfill{##1}}
    }{
      \par\noindent\makebox[0.15\columnwidth][l]{##2}{##3}
    }
  }
  \providecommand{\header}[3][\sht@null@arg]{
    \item[\textbf{##1}]{\textbf{##2}}{\textbf{##3}}
  }
}{
}
% ------------------------------------------------------------------------------
%   Bibliography
% ------------------------------------------------------------------------------
\citestyle{authoryear}
\bibliographystyle{gbt7714-plain}
\setlength{\bibsep}{0.0ex plus 0.2ex minus 0.2ex}
\newcommand{\bibstyle@authoryear}{\bibpunct{(}{)}{;}{a}{,}{,}}
\renewcommand*{\bibfont}{\small}
\newcommand{\makebiblio}[1][reference]{%
  \intotoc*{\cleardoublepage}{\bibname}%
  \bibliography{#1}%
}
% ------------------------------------------------------------------------------
%   Appendix
% ------------------------------------------------------------------------------
\LetLtxMacro{\origin@appendix}{\appendix}
\renewcommand{\appendix}{%
  \origin@appendix
  \intotoc*{\cleardoublepage}{\appendixname}%
  \settocdepth{part}%
  \pagestyle{LRPageNumberedAppendix}%
}
% ------------------------------------------------------------------------------
%   Backmatter
% ------------------------------------------------------------------------------
\providecommand{\backmatter}{}
\LetLtxMacro{\origin@backmatter}{\backmatter}
\renewcommand{\backmatter}{%
  \origin@backmatter%
  \settocdepth{chapter}%
  \pagestyle{LRPageNumbered}%
}
\newcounter{sht@resume@cnt}
\setcounter{sht@resume@cnt}{0}
\newcommand{\sht@check@resume@title}{
  \ifnum\thesht@resume@cnt=0
    \cleardoublepage
    \chapter{作者简历及攻读学位期间发表的学术论文与研究成果}
  \fi
  \stepcounter{sht@resume@cnt}
}
\newenvironment{resume}{%
  \sht@check@resume@title
  \section*{作者简历：}
  \ifsht@anonymous
    \sht@anonymous@str\par
    \comment
  \fi
}{%
  \ifsht@anonymous
    \endcomment
  \fi
}
\newenvironment{publications}{%
  \sht@check@resume@title
  \ifsht@anonymous
    \comment
  \fi
  \section*{已发表（或正式接受）的学术论文：}
}{%
  \ifsht@anonymous
    \endcomment
  \fi
}
\newenvironment{publications*}{%
  \sht@check@resume@title
  \ifthenelse{\not \boolean{sht@anonymous}}{
    \comment
  }{}
  \section*{已发表（或正式接受）的学术论文：}
}{%
  \ifthenelse{\not \boolean{sht@anonymous}}{
    \endcomment
  }{}
}
\renewenvironment{patterns}{%
  \sht@check@resume@title
  \ifsht@anonymous
    \comment
  \fi
  \section*{申请或已获得的专利：}
}{%
  \ifsht@anonymous
    \endcomment
  \fi
}
\newenvironment{patterns*}{%
  \sht@check@resume@title
  \ifthenelse{\not \boolean{sht@anonymous}}{
    \comment
  }{}
  \section*{申请或已获得的专利：}
}{%
  \ifthenelse{\not \boolean{sht@anonymous}}{
    \endcomment
  }{}
}
\newenvironment{projects}{%
  \sht@check@resume@title
  \section*{参加的研究项目及获奖情况：}
  \ifsht@anonymous
    \sht@anonymous@str\par
    \comment
  \fi
}{%
  \ifsht@anonymous
    \endcomment
  \fi
}
\newenvironment{acknowledgement}{%
  \ifsht@anonymous
    \comment
  \else
    \cleardoublepage
    \chapter[致谢]{致\hspace{1\ccwd}谢}\chaptermark{致\hspace{1\ccwd}谢}%
  \fi
}{%
  \ifsht@anonymous
    \endcomment
  \fi
}
% ==============================================================================
% Post-processing
% ==============================================================================
\AtEndDocument{
  \cleardoublepage
}
\endinput
