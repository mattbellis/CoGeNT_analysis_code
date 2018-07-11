# We pipe in "n" as an answer to --sep which wants to build
# separate build/source directories under doc
echo n | \
sphinx-quickstart \
--dot='_' \
--project='cogent' \
--author='Matt Bellis' \
-v='0.9' \
-r='0.9' \
-l='en' \
--suffix='.rst' \
--master='index' \
--epub \
--ext-autodoc  \
--ext-doctest  \
--ext-intersphinx  \
--ext-todo  \
--ext-coverage  \
--ext-imgmath  \
--ext-mathjax  \
--ext-ifconfig  \
--ext-viewcode  \
--use-make-mode \
--makefile \
--ext-githubpages \
--batchfile  \
doc 

sphinx-apidoc -o doc ./

cd doc

sed -i "18 i import sys" conf.py
sed -i "19 i sys.path.append('/home/amanda/CoGeNT_analysis_code')" conf.py

make html
