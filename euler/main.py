import math
from collections import defaultdict
import numpy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   s = "(hola)chino(chau)china"
   arr = s.split("(")
   for i in range(1, len(arr)):
      print(arr[i])

   arr = [[1,2], [3,4]]
   print(arr)

   d = {}
   for coordinates in arr:
      d[coordinates[0]] = coordinates[1]

