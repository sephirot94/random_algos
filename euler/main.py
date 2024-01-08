from leet.playing import Play
from leet.google import Google

def reverse_words(words: str) -> str:
    """Reverses the words of a given string and returns it. Words are separated by a whitespace"""
    word_list = words.split(" ")
    word_list = [word_list[i] for i in range(len(word_list)-1, -1, -1)]
    return " ".join(word_list)


if __name__ == '__main__':
    arr =[(0,1), (1,2), (0,3), (1,4), (1,3), (7,10)]
    goog = Google()
    print(goog.room_scheduling_conflicts(arr))