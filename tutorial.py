import sys

myvar = "Global var"

def show_size_of(obj):
    print obj.__class__, sys.getsizeof(obj)

def printloc():
    locvar = "Local var"
    global myvar
    print myvar
    myvar = "New my var"
    print myvar

