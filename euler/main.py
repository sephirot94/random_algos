import math
from collections import defaultdict
# from leet.google import Google

from trees.tree import Trie, AhoCorasick

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   # trie = Trie()
   # words = ["wait", "waiter", "shop", "shopify"]
   # for word in words:
   #    trie.insert(word)
   #
   # print(trie.search("salamander"))
   # print(trie.search("shopify"))
   # print(trie.search("waiter"))
   # print(trie.search("waitress"))
   # print(trie.search("shop"))
   # print("now we remove")
   #
   # trie.remove("shop")
   # trie.remove("waiter")
   # trie.remove("wai")
   # print(trie.search("shop"))
   # print(trie.search("shopify"))
   # print(trie.search("wait"))
   # print(trie.search("waiter"))

   x = AhoCorasick()
   x.add_word("he")
   x.add_word("she")
   x.add_word("his")
   x.add_word("hers")
   x.set_failure()
   x.display()

   a = "she is hers"

   x.find_string(a)