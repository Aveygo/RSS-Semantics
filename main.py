rss_feads = [
    {"name":"9 News", "url": "https://www.9news.com/feeds/syndication/rss/news"},
    {"name":"New York Times", "url": "https://content.api.nytimes.com/svc/news/v3/all/recent.rss"},
    {"name":"Independent", "url": "https://www.independent.co.uk/news/rss"},
    {"name":"The Guardian", "url": "https://www.theguardian.com/world/rss"},
    {"name":"Sbs", "url":"https://www.sbs.com.au/news/topic/latest/feed"},
    {"name":"Fox News", "url":"https://moxie.foxnews.com/google-publisher/world.xml"},
    {"name":"Wall Street", "url":"https://feeds.a.dj.com/rss/RSSWorldNews.xml"},
    {"name":"Forbes (Innovation)", "url":"https://www.forbes.com/innovation/feed"},
    {"name":"Forbes (Business)", "url":"https://www.forbes.com/business/feed"},
    {"name":"BBC", "url":"http://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name":"News.com", "url":"https://www.news.com.au/content-feeds/latest-news-world/"},
    {"name":"SMH", "url":"https://www.smh.com.au/rss/feed.xml"},
    {"name":"Herald Sun", "url":"https://www.heraldsun.com.au/rss"},
    {"name":"ABC", "url":"https://www.abc.net.au/news/feed/51120/rss.xml"},
    {"name":"The Sun Daily", "url":"https://www.thesundaily.my/rss/world"},
    {"name":"Washington Post", "url":"https://feeds.washingtonpost.com/rss/politics"},
    {"name":"Google Technology", "url":"https://rss.app/feeds/tN5PbWSGMXLukswD.xml"},
    {"name":"Google Business", "url":"https://rss.app/feeds/JlbwNUgQDLSbwFyp.xml"},
    {"name":"CNN", "url":"http://rss.cnn.com/rss/edition.rss"},
    {"name":"Investopedia", "url":"https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_articles"},
    {"name":"Reuters", "url":"https://rss.app/feeds/tt7Hgp6lgqECOntY.xml"},
    {"name":"Yahoo Finance", "url":"https://www.yahoo.com/news/rss"},
    {"name":"The Hill", "url":"https://thehill.com/news/feed/"},
    {"name":"The Economist", "url":"https://www.economist.com/the-world-this-week/rss.xml"},
    {"name":"The Atlantic", "url":"https://www.theatlantic.com/feed/all/"},
    {"name":"News Week", "url":"http://www.newsweek.com/rss"},
    {"name":"The Verge", "url":"https://www.theverge.com/rss/index.xml"},
    {"name":"Tech Crunch", "url":"https://techcrunch.com/feed/"},
    {"name":"Business Insider", "url":"http://feeds2.feedburner.com/businessinsider"},
    {"name":"China Daily", "url":"http://www.chinadaily.com.cn/rss/world_rss.xml"},
    {"name":"The Hindu", "url":"https://www.thehindu.com/news/feeder/default.rss"},
    {"name":"Global Times", "url":"http://www.globaltimes.cn/rss/outbrain.xml"},
    {"name":"CoinDesk", "url":"https://feeds.feedburner.com/CoinDesk"},
    {"name":"Fortune", "url":"http://fortune.com/feed/"}
]

import feedparser, numpy as np, zlib, base64, curses, time, threading, sys, os, logging, random, json, subprocess, subprocess, traceback
from datetime import datetime 
import torch
from transformers import BertTokenizer, BertModel

try:
    # Optional, adds clipboard functionality
    import pyperclip
except:
    pass

class RenderArticles:
    def __init__(self, stdscr, updater):
        self.stdscr = stdscr
        self.h, self.w = stdscr.getmaxyx()

        # Updater is threaded, updater.items is updated while running 
        self.updater = updater
        
        # Cursor vars; controls viewing window of items
        self.cursor = 1
        self.min_item_index = 0

        # Commands 
        self.center_text = "| 0:quit | 1:extract | 2:filter | 3:clear | 4:reset | 5:headlines | 6:shuffle | enter:copy | up/down:scroll |"
        
        # Current user query for news titles
        self.user_query = ""

        # The news items to query + display
        self.items = self.get_items()

        # Vector that user selects from an item
        # "a" controls mixing strength (0->ignore next vector, 0.3->weak, 1->ignore previous vector)
        self.user_vector = ""
        self.a = 0.3

        # Additional messages that any other object can control (mainly for debugging, other updates, etc)
        self.additional_message = f""
    
    def query(self, query):
        # Note, query is a lambda function
        for line in self.items:
            if query(line):
                yield line

    def query_by_score(self, query, reverse=False):
        # query is a lambda function that returns a float score
        data = []
        for line in self.items:
            score = query(line)
            if score: data.append((score, line))

        data.sort(key=lambda x: x[0], reverse=reverse)
        for score, line in data:
            yield line
    
    def get_items(self):
        # Return all items in the database
        self.items = self.updater.items
        self.updater.previously_got = len(self.items)

        # Convert compressed vectors to numpy arrays for each item
        for item in self.items:
            if type(item["vector"]) != np.ndarray:
                item["vector"] = self.vec_from_compresed(item["vector"])

        # Sort by publish date
        self.items = list(self.query_by_score(lambda x: x["published"], reverse=True))

        # Sort by query (number of same words)
        if self.user_query != "":
            sort_fn = lambda x: len([word for word in self.user_query.split(" ") if word in x["title"].lower()])
            return list(self.query_by_score(sort_fn, reverse=True))
        
        return self.items
        
    def time_to_human(self, epoch_time:int):
        # Convert epoch time to human readable time (e.g. 1 hour ago)
        diff = datetime.now() - datetime.fromtimestamp(epoch_time)
        if diff.days > 0: return f"{diff.days}d"
        elif diff.seconds > 3600: return f"{diff.seconds//3600}h"
        elif diff.seconds > 60: return f"{diff.seconds//60}m"
        else: return f"{diff.seconds}s"

    def vec_from_compresed(self, compressed:str):
        # Converts a compresed string from db to vector 
        vector = zlib.decompress(base64.b64decode(compressed))
        vector = np.frombuffer(vector, dtype=np.float16)
        return vector
    
    def vec_to_compresed(self, vector:np.ndarray):
        if vector == "": return ""
        # Converts a vector to a compresed string
        vector = vector.astype(np.float16)
        return base64.b64encode(zlib.compress(vector)).decode("utf-8")
    
    def vec_sim(self, a:np.ndarray, b:np.ndarray):
        # Euclidean distance between two vectors
        return 1 / (1 + np.linalg.norm(a - b))

    def kmeans(self, vectors, num_clusters=3, max_iterations=50):
        # Attempt to find clusters in a list of vectors
        agents = []
        for agent in range(num_clusters):
            agents.append(vectors[np.random.randint(0, len(vectors))])

        for iteration in range(max_iterations):
            clusters = [[] for agent in range(num_clusters)]
            for vector in vectors:
                distances = []
                for agent in agents:
                    distances.append(np.linalg.norm(vector - agent))
                clusters[distances.index(min(distances))].append(vector)
            
            for agent in range(num_clusters):
                agents[agent] = np.mean(clusters[agent], axis=0)
                    
        return agents

    def render(self):
        ## Begin base layout
        # Show user vector (top right)
        self.stdscr.addstr(0, self.w-50, self.vec_to_compresed(self.user_vector)[:50], curses.color_pair(2))
        # Show key commands (bottom middle)
        self.stdscr.addstr(self.h-1, self.w//2 - len(self.center_text)//2, self.center_text)
        # Show search bar (top left) and query
        self.stdscr.addstr(0, 0, "Search: ", curses.color_pair(2))
        self.stdscr.addstr(0, 8, self.user_query, curses.color_pair(3))

        # Found n articles is updated in GetItems, 

        # Check if items exist in the database
        self.db_size = len(self.items)
        if not self.db_size:
            self.stdscr.addstr(1, 0, "Empty query or database is being loaded (press 4 to try again)...")
        
        # Draw articles
        for i, item in enumerate(self.items[self.min_item_index:self.min_item_index+self.h-2]):

            # Cleaning the source
            source = item["source"] if not item["source"] is None else "unknown"
            if len(source) > 15: source = source[:12] + "..."
            source += " " * (15 - len(source))

            # Cleaning published time
            published = self.time_to_human(item["published"])
            published += " " * (5 - len(published))

            # Adding the title
            title = "--- No title ---" if item["title"] == "" else item["title"]
            if len(title) > self.w-10: title = title[:self.w-10] + "..."

            # Combining the line
            line = f"{source} | {published} | {title}"
            
            # Change color based on if cursor is on it
            if self.cursor == i+1: self.stdscr.addstr(i+1, 0, line, curses.color_pair(1))
            else: self.stdscr.addstr(i+1, 0, line)
        
        # Grab key presses
        key = self.stdscr.getch()

        # Move down
        if key == curses.KEY_DOWN:
            if self.cursor == self.h-2: self.min_item_index += 1
            else: self.cursor += 1

        # Move up
        elif key == curses.KEY_UP:
            if self.cursor == 1: self.min_item_index -= 1
            else: self.cursor -= 1
        
        # Exit button
        elif key == ord("0"):
            return {"running":False}

        # Grab item vector and "add" it to the user vector
        elif key == ord("1"):
            item_vector = self.items[self.min_item_index:self.min_item_index+self.h-2][self.cursor-1]["vector"]
            if self.user_vector == "": self.user_vector = item_vector
            self.user_vector = self.a * item_vector + (1-self.a) * self.user_vector
        
        # Prune items that are not similar to the user vector
        elif key == ord("2"):

            # Ignore if user vector is empty
            if self.user_vector == "": return {"running":True}

            # Prune by date if too large
            items = self.items[:1000]
            
            # Reset item scores
            for query_item in self.items:
                query_item["score"] = 0

            # Calculate scores
            for query_item in self.items:
                score = self.vec_sim( query_item["vector"], self.user_vector )
                query_item["score"] = score
                        
            # Sort by score
            self.items = sorted(self.items, key=lambda x: x["score"], reverse=True)

            # Prune half of the items
            self.items = self.items[:len(self.items)//2]
            
            # Reset cursors
            self.cursor = 1
            self.min_item_index = 0
        
        # Reset user vector
        elif key == ord("3"):
            self.user_vector = ""
        
        # Reset all items, queries, user vector
        elif key == ord("4"):
            self.cursor = 1
            self.min_item_index = 0
            self.items = self.get_items()
        
        # Attempt to find headlines
        elif key == ord("5"):
            # Prune by date if too large
            items = self.items[:500]

            # Attempt to find clusters (common titles probably means major event with lots of articles)
            headline_vectors = self.kmeans([item["vector"] for item in items])

            # Reset scores
            for query_item in items:
                query_item["score"] = 0

            # Find the most similar vectors to the found centers of clusters
            for agent in headline_vectors:
                for item in items:
                    score = self.vec_sim( item["vector"], agent )
                    item["score"] += score
            
            # Set items to found headlines and reset cursor
            self.items = sorted(items, key=lambda x: x["score"], reverse=True)
            self.cursor = 1
            self.min_item_index = 0

        # Suffle items
        elif key == ord("6"):
            self.cursor = 1
            self.min_item_index = 0
            random.shuffle(self.items)

        # Remove characters from the query
        elif key == curses.KEY_BACKSPACE:
            self.items = self.get_items()
            self.user_query = self.user_query[:-1]
        
        # Copy the selected article url to clipboard
        elif key == ord("\n"):
            item_url = self.items[self.min_item_index:self.min_item_index+self.h-2][self.cursor-1]["link"]
            try:
                # Check if pyperclip was installed and imported 
                pyperclip
            except:
                raise Exception(f"Please run '{sys.executable} -m pip install pyperclip' to use the paperclip functionality")

            pyperclip.copy(item_url)

        # Add characters to the query & reset cursor
        elif key != ord("\n"):
            self.user_query += chr(key)
            self.items = self.get_items()
            self.min_item_index = 0
            self.cursor = 1

        # Limit cursor and min_item_index to valid values
        self.min_item_index = max(min(self.min_item_index, self.db_size), 0)
        self.cursor = max(min(self.cursor, self.h-2), 1)

        return {"running":True}

class GetItems:
    def __init__(self, stdscr):
        # Updates the database for news items
        self.update_every = 60 * 10 # 10 minutes
        self.running = True
        self.last_update = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.items = []
        self.previously_got = 0

        self.h, self.w = stdscr.getmaxyx()
        self.stdscr = stdscr

    def sentence2vector(self, sentence):
        # Converts a sentence to a vector
        marked_text = "[CLS] " + sentence.lower() + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = self.model(tokens_tensor.to(self.device), segments_tensors.to(self.device))
            hidden_states = outputs[2]
            token_vecs = hidden_states[-2][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)
            return sentence_embedding.cpu().numpy()
    
    def run(self):
        # Load model
        logging.disable(logging.CRITICAL)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).eval().to(self.device)

        # Updates the db every update_every seconds
        while self.running:
            if time.time() - self.last_update > self.update_every:
                self.last_update = time.time()
                self.update_db()   
            else:
                time.sleep(1)             

    def get_representation(self, text:str):
        # To store the vector, it needs to be converted to a string
        vector = self.sentence2vector(text)
        vector = np.array(vector, dtype=np.float16)
        return base64.b64encode(zlib.compress(vector)).decode("utf-8")
    
    def update_db(self):
        for i, feed in enumerate(rss_feads):

            # Extract news items            
            for item in feedparser.parse(feed["url"])["items"]:
                
                # Check if the database already has an item with the same link
                if not len(list([x for x in self.items if x["link"] == item["link"]])):

                    # Convert readable published time to epoch for sorting 
                    published = time.time()
                    if "published" in item:
                        try: published = datetime.strptime(" ".join(item["published"].split(" ")[:-1]), '%a, %d %b %Y %H:%M:%S').timestamp()
                        except: pass

                    self.items.append({
                        "title":item["title"],
                        "link":item["link"],
                        "source":feed["name"], 
                        "published":published,
                        "vector":self.get_representation(item["title"])
                    })

                    if not self.running:
                        return
                    
                    if self.previously_got < len(self.items):
                        message = f"{len(self.items)-self.previously_got} new items found"
                        self.stdscr.addstr(self.h-1, 0, message)
                        self.stdscr.refresh()
        
        self.stdscr.addstr(self.h-1, 0, f"{message} - Done Loading!")
        self.stdscr.refresh()
                    

class Screen:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.h, self.w = stdscr.getmaxyx()

        # Start the updater
        self.updater = GetItems(stdscr)
        self.updater_thread = threading.Thread(target=self.updater.run)
        self.updater_thread.start()
        
        # Start colors, hde cursor
        curses.start_color()
        curses.curs_set(0)
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)

        stdscr.clear()
        stdscr.refresh()

        # Select a page to render in the main loop (must return state["running"]==True) 
        self.page = RenderArticles(stdscr, self.updater).render
    
        self.running = True
        self.run()

    def run(self):
        state = {"running":True}
        while self.running:
            self.stdscr.clear()

            try:
                state = self.page()
            except Exception as e:
                self.stdscr.clear()
                
                self.stdscr.addstr(0, 0, "Press any key to continue or q to quit")
                self.stdscr.addstr(2, 0, str(e))
                
                self.stdscr.refresh()
                state["running"] = self.stdscr.getch() != ord("q")

            if "switch_to" in state:
                self.page = state["switch_to"]
            
            self.running = state["running"]
            self.stdscr.refresh()

        self.stdscr.clear()
        self.stdscr.refresh()

        self.stdscr.addstr(0, 0, "Cleaning & Closing...")
        self.stdscr.refresh()

        self.updater.running = False
        self.updater_thread.join()        

        traceback.print_exc()

if __name__ == "__main__":
    curses.wrapper(Screen)
