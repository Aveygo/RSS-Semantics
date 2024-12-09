# ⚠️ Moved to [AutoInt](https://github.com/Aveygo/AutoInt)
Use [SemanticNews](https://github.com/Aveygo/SemanticNews)

Query news articles, discover headlines, filter by topic; For any rss feed, as quickly as possible.

![Sample photo](https://raw.githubusercontent.com/Aveygo/RSS-Semantics/main/sample.png "Logo Title Text 1")

### Note!
Only tested on Linux

No database, found articles are only stored in memory

## Steps to run:
1. Install libraries
(pytorch with cuda from [here](https://pytorch.org/get-started/locally/))
```
python3 -m pip install pyperclip numpy feedparser transformers
```
2. Dowload this [repo](https://github.com/Aveygo/RSS-Semantics/archive/refs/heads/main.zip)
3. Unzip and run with python! (tested on 3.7+)
```
python3 main.py
```

## Additional
By default there are 34 rss sources, but you can add your own by editing the "main.py" file and adding your own entry. Follow the syntax by adding a soure name and valid rss url. 

## Commands
0 : Exit the script

1 : Add semantics from selected title to user selection

2 : Remove and sort headlines by selected semantics

3 : Clear semantics selection

4 : Reset to default state and update newly found items

5 : Find headlines and sort by probability (rerunning will give different results)

6 : Randomly shuffle the current items

Enter : Copy the url to the clipboard

Up/Down : Select and scroll to different news articles

Others : Adds the character to the query for specific headlines

## Todo
Add multithreading support to download rss feeds

Add a database?
