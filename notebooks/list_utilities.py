import numpy as np


def match_sorted(arr1, arr2, index1=False, index2=False, verbose=0, key1=None, key2=None):
    """
    Find common elements on two sorted lists: O(n + m)

    Parameters
    ----------
    arr1, arr2 : array_like
        Two sorted arrays to operate on.
    index1 : bool
        Return the indices of matching elements for arr1.
    index2 : bool
        Return the indices of matching elements for arr2.
    verbose : (0, 1)
        If 1, prints information about the matching.
    key1, key2 : function
        If key is passed, the comparison will be made based on the return value of the key.

    Notes
    -----
    If the arrays contain duplicates, then the first matching duplicates are returned as
    index1 and index2. As an example:
    >>> arr1 = [1,1,1,2,3]
    >>> arr2 = [1,1,3,4]
    >>> match_sorted(arr1, arr2, index1=True, index2=True)
    (array([1, 1, 3]), array([0, 1, 4]), array([0, 1, 2]))
    """

    indices1 = []
    indices2 = []
    matching = []
    i = 0
    j = 0
    while (i < len(arr1) and j < len(arr2)):
        comp1 = arr1[i] if key1 is None else key1(arr1[i])
        comp2 = arr2[j] if key2 is None else key2(arr2[j])
        if (comp1 == comp2):
            indices1.append(i)
            indices2.append(j)
            matching.append(arr1[i])
            i += 1
            j += 1
        elif (comp1 > comp2):
            j += 1
        else:
            i += 1
    if (verbose == 1):
        print("List 1: {}, List 2: {}, Matching: {}".format(len(arr1), len(arr2), len(matching)))

    if (index1 and index2):
        return np.array(matching), np.array(indices1), np.array(indices2)
    elif (index1):
        return np.array(matching), np.array(indices1)
    elif (index2):
        return np.array(matching), np.array(indices2)

    return np.array(matching)


def make_unique(arr, return_sorted=False, return_index=False):
    """
    Modify non-unique elements of the string array `arr` to make them unique.

    Parameters
    ----------
    arr : array_like
        Array to operate on.
    return_sorted : bool, default=False
        Whether arr is returned sorted.
    return_index : bool, default=False
        Whether we return indices to sort/unsort the array. If arr is returned sorted, then
        the returned indices will unsort it, and vice-versa.

    Examples
    --------
    >>> arr = ["a", "b", "d", "a"]
    >>> make_unique(arr)
    ['a_1', 'b', 'd', 'a_2']
    >>> make_unique(arr, return_sorted=True)
    ['a_1', 'a_2', 'b', 'd']
    >>> make_unique(arr, return_index=True)
    ['a_1', 'b', 'd', 'a_2'], [0, 3, 1, 2]
    >>> make_unique(arr, return_sorted=True, return_index=True)
    ['a_1', 'a_2', 'b', 'd'], [0, 2, 3, 1]

    Notes
    -----
    This function makes calls to `sorted` and `np.argsort`, and therefore works O(nlogn) at best.
    """

    sort_idx = np.argsort(arr, kind="stable")
    arr = sorted(arr)

    count = 1
    current = 0
    for name in arr[1:]:
        if arr[current] == name:
            arr[current] = f"{arr[current]}_{count}"
            count += 1
        else:
            if count > 1:
                arr[current] = f"{arr[current]}_{count}"
            count = 1
        current += 1
    if count > 1:
        arr[current] = f"{arr[current]}_{count}"

    arr = np.array(arr)
    resort_idx = np.argsort(sort_idx, kind="stable")

    if return_sorted:
        if return_index:
            return arr, resort_idx
        return arr

    if return_index:
        return arr[resort_idx], sort_idx

    return arr[resort_idx]


def make_unique_fast(arr):
    """
    Modify non-unique elements of the string array `arr` to make them unique.

    Notes
    -----
    Unlike `make_unique`, this function works in linear time.
    """

    if type(arr) == np.ndarray and np.issubdtype(arr.dtype, np.str_):
        arr = arr.astype(np.object_)  # cast string dtype to object

    d = {}
    for elem in arr:
        try:
            d[elem] += 1
        except KeyError:
            d[elem] = 1

    for key, value in d.items():
        if value == 1:
            d[key] = 0

    for i in range(len(arr) - 1, -1, -1):
        elem = arr[i]
        if d[elem] > 0:
            arr[i] = f"{arr[i]}_{d[elem]}"
            d[elem] -= 1

    return arr


def similarize_class_ratios(x, y, err_on_zero=False, one_on_zero=True):
    """
    Finds the indices in ``x``, that will produce the same class ratio as in ``y``.

    Returns
    -------
    The indices from ``x`` to keep.
    """
    ratio = np.count_nonzero(y == 0) / np.count_nonzero(y == 1)
    return reduce_class_ratios(x, ratio, err_on_zero, one_on_zero)


def reduce_class_ratios(x, ratio, err_on_zero=False, one_on_zero=True):
    """
    Given a ratio of class 0 to class 1, find the indices to keep so that
    the ratio of class 0 to class 1 is equal to ``ratio`` for x.

    If the data is too small in size, then the given ratio might fail to include any
    samples from one of the classes. In such cases, indices for the entire ``x`` is returned.
    Check ``err_on_zero`` and ``one_on_zero`` to control this behavior.

    Parameters
    ----------
    x : array_like
        Array to operate on.
    ratio : float
        Ratio of class 0 to class 1
    err_on_zero : bool
        If ``err_on_zero`` is True, we throw an error instead of returning all class indices.
    one_on_zero : bool
        If ``one_on_zero`` is True and ``err_on_zero`` is False,
        we return one sample from the zero-ed class instead of returning all class indices.

    Returns
    -------
    Indices to keep to maintain the ratio.
    """
    class0_idx = np.where(x == 0)[0]
    class1_idx = np.where(x == 1)[0]
    num_class0 = np.count_nonzero(x == 0)
    num_class1 = np.count_nonzero(x == 1)

    needed_0 = int(num_class1 * ratio)
    needed_1 = int(num_class0 / ratio)

    if needed_0 == 0 or needed_1 == 0:
        if err_on_zero:
            raise ValueError("Data is too small for the ratio. Can't produce meaningful data.")
        elif one_on_zero:
            if needed_0 == 0:
                return np.concatenate((class0_idx[0:1], class1_idx), axis=0)
            else:
                return np.concatenate((class0_idx, class1_idx[0:1]), axis=0)
        else:
            return np.concatenate((class0_idx, class1_idx), axis=0)

    if needed_0 <= num_class0:
        class0_idx = class0_idx[0:needed_0]
    elif needed_1 <= num_class1:
        class1_idx = class1_idx[0:needed_1]
    else:
        raise Exception(f"How did I get here? ratio:{ratio}, needed_0:{needed_0}, needed_1:{needed_1}, num_class0:{num_class0}, num_class1:{num_class1}")

    return np.concatenate((class0_idx, class1_idx), axis=0)
