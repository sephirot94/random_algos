from collections import defaultdict


class KMPPatternSearch:
    """
    This algorithm finds all occurrences of a pattern in a text in linear time. Let length of text be n
    and of pattern be m, then total time taken is O(m + n) with linear space complexity
    """
    def __init__(self):
        pass

    def gen_lps_array(self, pattern: str, M: int, lps: list):
        """
        Generates the lps array used by KMP algorithm
        """
        length = 0  # length of the previous longest prefix suffix
        lps[0] = 0  # always first element is 0
        i = 1
        while i < M:  # calculate lps[i] for i-1 to M-1
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length-1]
                    # do not increment 1 here
                else:
                    lps[i] = 0
                    i += 1


    def search(self, pattern: str, text: str) -> set:
        M = len(pattern)
        N = len(text)
        pat_idx_set = set()
        lps = [0 for i in range(M)]  # lps[] will hold the longest prefix suffix values for pattern
        j = 0  # index for pattern[]
        self.gen_lps_array(pattern, M, lps)  # generate lps array

        i = 0  # index for text
        while i < N:  # search text
            if pattern[j] == text[i]:
                i += 1
                j += 1
            if j == M:
                print(f"Found pattern at index {str(i-j)}")
                pat_idx_set.add(i-j)
                j = lps[j-1]
            elif i < N and pattern[j] != text[i]:  # mismatch after j matches
                if j != 0:  # no need to match lps[0..lps[j-1]] since they already match
                    j = lps[j-1]
                else:
                    i += 1
        return pat_idx_set


class FiniteAutomataPatternSearching:

    NUMBER_CHARS = 256

    def __init__(self):
        pass

    def compute_automata(self, pattern: str, M: int, TF: list):  # O(M*NO_OF_CHARS)
        lps = 0
        for x in range(self.NUMBER_CHARS):
            TF[0][x] = 0

        TF[0][ord(pattern[0])] = 1
        for i in range(1, M+1):
            for x in range(self.NUMBER_CHARS):
                TF[i][x] = TF[lps][x]

            if i < M:
                TF[i][ord(pattern[i])] = i + 1  # Update the entry of this character
                lps = TF[lps][ord(pattern[i])]  # Update lps for next row to be filled

    def search(self, pattern: str, txt: str) -> set:  # O(N)
        M = len(pattern)
        N = len(txt)
        resp = set()
        TF = [[0 for i in range(self.NUMBER_CHARS)] for j in range(M + 1)]
        self.compute_automata(pattern, M, TF)

        # process text over FA.
        j = 0
        for i in range(N):
            j = TF[j][ord(txt[i])]
            if (j == M):
                print("pattern found at index", i - M + 1)
                resp.add(i-M+1)

        return resp


class AhoCorasickPatternSearching:
    """
    Given an input text and an array of k words, arr[], find all occurrences of all words in the input text.
    Let n be the length of text and m be the total number characters in all words, i.e. m = length(arr[0]) +
    length(arr[1]) + … + length(arr[k-1]). Here k is total numbers of input words.
    """
    # Aho-Corasick Algorithm finds all words in O(n + m + z) time where z is total number of occurrences of words
    def __init__(self, words):

        # Max number of states in the matching machine. Should be equal to the sum of the length of all keywords.
        self.max_states = sum([len(word) for word in words])

        # Maximum number of characters. Currently supports only alphabets [a,z]
        self.max_characters = 26

        # OUTPUT FUNCTION IS IMPLEMENTED USING out []
        # Bit i in this mask is 1 if the word with
        # index i appears when the machine enters this state.
        # Lets say, a state outputs two words "he" and "she" and
        # in our provided words list, he has index 0 and she has index 3
        # so value of out[state] for this state will be 1001
        # It has been initialized to all 0.
        # We have taken one extra state for the root.
        self.out = [0] * (self.max_states + 1)

        # FAILURE FUNCTION IS IMPLEMENTED USING fail []
        # There is one value for each state + 1 for the root
        # It has been initialized to all -1
        # This will contain the fail state value for each state
        self.fail = [-1] * (self.max_states + 1)

        # GOTO FUNCTION (OR TRIE) IS IMPLEMENTED USING goto [[]]
        # Number of rows = max_states + 1
        # Number of columns = max_characters i.e 26 in our case
        # It has been initialized to all -1.
        self.goto = [[-1] * self.max_characters for _ in range(self.max_states + 1)]

        # Convert all words to lowercase
        # so that our search is case insensitive
        for i in range(len(words)):
            words[i] = words[i].lower()

        # All the words in dictionary which will be used to create Trie
        # The index of each keyword is important:
        # "out[state] & (1 << i)" is > 0 if we just found word[i]
        # in the text.
        self.words = words

        # Once the Trie has been built, it will contain the number
        # of nodes in Trie which is total number of states required <= max_states
        self.states_count = self.__build_matching_machine()

    # Builds the String matching machine.
    # Returns the number of states that the built machine has.
    # States are numbered 0 up to the return value - 1, inclusive.
    def __build_matching_machine(self):
        k = len(self.words)

        # Initially, we just have the 0 state
        states = 1

        # Convalues for goto function, i.e., fill goto
        # This is same as building a Trie for words[]
        for i in range(k):
            word = self.words[i]
            current_state = 0

            # Process all the characters of the current word
            for character in word:
                ch = ord(character) - 97  # Ascii valaue of 'a' = 97

                # Allocate a new node (create a new state)
                # if a node for ch doesn't exist.
                if self.goto[current_state][ch] == -1:
                    self.goto[current_state][ch] = states
                    states += 1

                current_state = self.goto[current_state][ch]

            # Add current word in output function
            self.out[current_state] |= (1 << i)

        # For all characters which don't have
        # an edge from root (or state 0) in Trie,
        # add a goto edge to state 0 itself
        for ch in range(self.max_characters):
            if self.goto[0][ch] == -1:
                self.goto[0][ch] = 0

        # Failure function is computed in
        # breadth first order using a queue
        queue = []

        # Iterate over every possible input
        for ch in range(self.max_characters):

            # All nodes of depth 1 have failure
            # function value as 0. For example,
            # in above diagram we move to 0
            # from states 1 and 3.
            if self.goto[0][ch] != 0:
                self.fail[self.goto[0][ch]] = 0
                queue.append(self.goto[0][ch])

        # Now queue has states 1 and 3
        while queue:

            # Remove the front state from queue
            state = queue.pop(0)

            # For the removed state, find failure
            # function for all those characters
            # for which goto function is not defined.
            for ch in range(self.max_characters):

                # If goto function is defined for
                # character 'ch' and 'state'
                if self.goto[state][ch] != -1:

                    # Find failure state of removed state
                    failure = self.fail[state]

                    # Find the deepest node labeled by proper
                    # suffix of String from root to current state.
                    while self.goto[failure][ch] == -1:
                        failure = self.fail[failure]

                    failure = self.goto[failure][ch]
                    self.fail[self.goto[state][ch]] = failure

                    # Merge output values
                    self.out[self.goto[state][ch]] |= self.out[failure]

                    # Insert the next level node (of Trie) in Queue
                    queue.append(self.goto[state][ch])

        return states

    # Returns the next state the machine will transition to using goto
    # and failure functions.
    # current_state - The current state of the machine. Must be between
    #             0 and the number of states - 1, inclusive.
    # next_input - The next character that enters into the machine.
    def __find_next_state(self, current_state, next_input):
        answer = current_state
        ch = ord(next_input) - 97  # Ascii value of 'a' is 97

        # If goto is not defined, use
        # failure function
        while self.goto[answer][ch] == -1:
            answer = self.fail[answer]

        return self.goto[answer][ch]

    # This function finds all occurrences of all words in text.
    def search(self, text):
        # Convert the text to lowercase to make search case insensitive
        text = text.lower()

        # Initialize current_state to 0
        current_state = 0

        # A dictionary to store the result.
        # Key here is the found word
        # Value is a list of all occurrences start index
        result = defaultdict(list)

        # Traverse the text through the built machine
        # to find all occurrences of words
        for i in range(len(text)):
            current_state = self.__find_next_state(current_state, text[i])

            # If match not found, move to next state
            if self.out[current_state] == 0:
                continue

            # Match found, store the word in result dictionary
            for j in range(len(self.words)):
                if (self.out[current_state] & (1 << j)) > 0:
                    word = self.words[j]

                    # Start index of word is (i-len(word)+1)
                    result[word].append(i - len(word) + 1)

        # Return the final result dictionary
        return result
