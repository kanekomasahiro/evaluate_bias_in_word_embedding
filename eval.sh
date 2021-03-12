wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
unzip wiki-news-300d-1M-subword.vec.zip
mkdir -p output
embedding=wiki-news-300d-1M-subword.vec

echo WEAT
python -u weat.py --embedding $embedding --output output/weat_${embedding}.txt
echo WAT
python -u wat.py --embedding $embedding --output output/wat_${embedding}.txt
echo SemBias
python -u sembias.py --embedding $embedding --output output/sembias_${embedding}.txt
