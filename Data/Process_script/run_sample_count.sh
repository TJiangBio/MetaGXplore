conda init bash
conda activate py37

rdfind -deleteduplicates true ./

less *cna*TM* | cut -f 1 | sort -u > 1
less -S *meth*TM* | cut -f 5 | head -n 1 | cat >>1
less -S *TM*bcgsc* | cut -f 16 | sort -u | cat >>1
less -S *gene.normalized*TM* | head -n 1 | sed 's/\t/\n/g' | sed 1d | sort -u | cat >>1
echo"##################sample number is##################"
less 1| sort -u | wc


