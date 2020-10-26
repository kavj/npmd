from graphviz import Source

p = "while.dot" 
s = Source.from_file(p)
s.view()


