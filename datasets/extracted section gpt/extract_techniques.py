import pandas as pd
with open("corpus.txt","r") as f:
      l=f.read().replace("- ","").split("\n")
with open("all_techniques.txt","w") as f:
      f.write("\n".join(list(set(l))))
