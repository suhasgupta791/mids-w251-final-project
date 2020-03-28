#!/usr/bin/python

import sys, getopt
import praw
from praw.models import MoreComments
import json

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:",["subreddit="])
   except getopt.GetoptError:
      print 'test.py -s <subreddit>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'test.py -s <subreddit>'
         sys.exit()
      elif opt in ("-s", "--subreddit"):
         inputfile = arg




if __name__ == "__main__":
   main(sys.argv[1:])

