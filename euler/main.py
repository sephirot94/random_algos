import math


def max_number_lectures(arrival, duration):
    """
    Given an array contianing time of arrival and another array containing duration of stay,
    determine how many lectures can be made in single room without two occuring at same time
    :param arrival: array containing time of arrival for each participant
    :param duration: array containing duration of each participant's lecture
    :return: max number of lectures that can occur in a single day
    """
    # O(nlogn) Time
    ans = 0

    # Sorting of meeting according to
    # their finish time.
    zipped = zip(arrival, duration)
    zipped = list(zipped)
    zipped.sort(key=lambda x: x[0] + x[1])

    # Initially select first meeting
    ans += 1
    # time_limit to check whether new
    # meeting can be conducted or not.
    time_limit = zipped[0][0] + zipped[0][1]

    # Check for all meeting whether it
    # can be selected or not.
    for i in range(1, len(arrival)):
        if zipped[i][0] > time_limit:
            ans += 1
            time_limit = zipped[i][0] + zipped[i][1]

    return ans

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arrival = [978, 409, 229, 934, 299, 982, 636, 14, 866, 815, 64, 537, 426, 670, 116, 95, 630]
    duration = [502, 518, 196, 106, 405, 452, 299, 189, 124, 506, 883, 753, 567, 717, 338, 439, 145]
    print(max_number_lectures(arrival, duration))