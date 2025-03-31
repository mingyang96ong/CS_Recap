import pytest

from strings import *

from random import randint
from copy import deepcopy

@pytest.fixture
def convert_base_testcases():
    test_cases = [
        # Convert base 2 to base 10
        (("1010", 2, 10), "10"),      # "1010" in base 2 is "10" in base 10
        (("111", 2, 10), "7"),        # "111" in base 2 is "7" in base 10
        (("100101", 2, 10), "37"),    # "100101" in base 2 is "37" in base 10

        # Convert base 10 to base 2
        (("10", 10, 2), "1010"),      # "10" in base 10 is "1010" in base 2
        (("7", 10, 2), "111"),        # "7" in base 10 is "111" in base 2
        (("37", 10, 2), "100101"),    # "37" in base 10 is "100101" in base 2

        # Convert base 16 to base 10
        (("A", 16, 10), "10"),        # "A" in base 16 is "10" in base 10
        (("F", 16, 10), "15"),        # "F" in base 16 is "15" in base 10
        (("1F", 16, 10), "31"),       # "1F" in base 16 is "31" in base 10

        # Convert base 10 to base 16
        (("10", 10, 16), "A"),        # "10" in base 10 is "A" in base 16
        (("15", 10, 16), "F"),        # "15" in base 10 is "F" in base 16
        (("31", 10, 16), "1F"),       # "31" in base 10 is "1F" in base 16

        # Convert base 8 to base 10
        (("12", 8, 10), "10"),        # "12" in base 8 is "10" in base 10
        (("17", 8, 10), "15"),        # "17" in base 8 is "15" in base 10
        (("24", 8, 10), "20"),        # "24" in base 8 is "20" in base 10

        # Convert base 10 to base 8
        (("10", 10, 8), "12"),        # "10" in base 10 is "12" in base 8
        (("15", 10, 8), "17"),        # "15" in base 10 is "17" in base 8
        (("20", 10, 8), "24"),        # "20" in base 10 is "24" in base 8

        # Edge cases (e.g., base 1 to base 2 or similar odd cases)
        (("1000", 2, 2), "1000"),     # "1000" in base 2 is still "1000" in base 2
        (("0", 10, 2), "0"),          # "0" in base 10 is "0" in base 2
        (("0", 10, 16), "0"),         # "0" in base 10 is "0" in base 16

        # Convert large numbers
        (("9876543210", 10, 2), "1001001100101100000001011011101010"),  # large base 10 to base 2
        (("1001001100101100000001011011101010", 2, 10), "9876543210"),  # large base 2 to base 10
    ]
    return test_cases

@pytest.fixture
def look_and_say_testcases():
    test_cases = [
        # Test case 1: The first term in the sequence
        ((1), "1"),  # The first term is simply "1"
        
        # Test case 2: The second term in the sequence
        ((2), "11"),  # The second term describes the first term "1", which is "one 1" -> "11"
        
        # Test case 3: The third term in the sequence
        ((3), "21"),  # The third term describes the second term "11", which is "two 1s" -> "21"
        
        # Test case 4: The fourth term in the sequence
        ((4), "1211"),  # The fourth term describes the third term "21", which is "one 2, one 1" -> "1211"
        
        # Test case 5: The fifth term in the sequence
        ((5), "111221"),  # The fifth term describes the fourth term "1211", which is "one 1, one 2, two 1s" -> "111221"
        
        # Test case 6: The sixth term in the sequence
        ((6), "312211"),  # The sixth term describes the fifth term "111221", which is "three 1s, two 2s, one 1" -> "312211"
        
        # Test case 7: The seventh term in the sequence
        ((7), "13112221"),  # The seventh term describes the sixth term "312211", which is "one 3, one 1, two 2s, two 1s" -> "13112221"
        
        # Test case 8: The eighth term in the sequence
        ((8), "1113213211"),  # The eighth term describes the seventh term "13112221", which is "one 1, one 3, two 1s, three 2s, one 1" -> "1113213211"
        
        # Test case 9: Checking for larger n (e.g., 10th term)
        ((10), "13211311123113112211"),  # The tenth term describes the ninth term "1113213211", which is "one 1, one 3, two 1s, one 1, one 2, three 1s, one 3, two 1s, one 2" -> "13211311123113112211"
        
        # Test case 10: Edge case of zero (not defined for the sequence, we may not support it in this problem)
        ((0), ""),  # There is no zero-th term in the Look-and-Say sequence

        # Test case 11: Very large case
        ((30), "3113112221131112311332111213122112311311123112111331121113122112132113121113222112311311221112131221123113112221121113311211131122211211131221131211132221121321132132212321121113121112133221123113112221131112212211131221121321131211132221123113112221131112311332211211133112111311222112111312211311123113322112111312211312111322212321121113121112133221121321132132211331121321132213211231132132211211131221232112111312212221121123222112311311222113111231133211121321321122111312211312111322211213211321322123211211131211121332211231131122211311123113321112131221123113111231121123222112111331121113112221121113122113111231133221121113122113121113221112131221123113111231121123222112111312211312111322212321121113121112131112132112311321322112111312212321121113122122211211232221121321132132211331121321231231121113112221121321132132211322132113213221123113112221133112132123222112111312211312112213211231132132211211131221131211322113321132211221121332211231131122211311123113321112131221123113111231121113311211131221121321131211132221123113112211121312211231131122211211133112111311222112111312211312111322211213211321223112111311222112132113213221133122211311221122111312211312111322212321121113121112131112132112311321322112111312212321121113122122211211232221121321132132211331121321231231121113112221121321132132211322132113213221123113112221133112132123222112111312211312112213211231132132211211131221131211322113321132211221121332211213211321322113311213212312311211131122211213211331121321123123211231131122211211131221131112311332211213211321223112111311222112132113213221123123211231132132211231131122211311123113322112111312211312111322111213122112311311123112112322211213211321322113312211223113112221121113122113111231133221121321132132211331222113321112131122211332113221122112133221123113112221131112311332111213122112311311123112111331121113122112132113121113222112311311221112131221123113112221121113311211131122211211131221131211132221121321132132212321121113121112133221123113112221131112311332111213122112311311123112112322211322311311222113111231133211121312211231131112311211232221121113122113121113222123211211131221132211131221121321131211132221123113112211121312211231131122113221122112133221121321132132211331121321231231121113121113122122311311222113111231133221121113122113121113221112131221123113111231121123222112132113213221133112132123123112111312211322311211133112111312211213211311123113223112111321322123122113222122211211232221121113122113121113222123211211131211121311121321123113213221121113122123211211131221121311121312211213211321322112311311222113311213212322211211131221131211221321123113213221121113122113121113222112131112131221121321131211132221121321132132211331121321232221123113112221131112311322311211131122211213211331121321122112133221121113122113121113222123112221221321132132211231131122211331121321232221121113122113121113222123211211131211121332211213111213122112132113121113222112132113213221232112111312111213322112132113213221133112132123123112111311222112132113311213211221121332211231131122211311123113321112131221123113112221132231131122211211131221131112311332211213211321223112111311222112132113212221132221222112112322211211131221131211132221232112111312111213111213211231131112311311221122132113213221133112132123222112311311222113111231132231121113112221121321133112132112211213322112111312211312111322212321121113121112131112132112311321322112111312212321121113122122211211232221121311121312211213211312111322211213211321322123211211131211121332211213211321322113311213211322132112311321322112111312212321121113122122211211232221121321132132211331121321231231121113112221121321133112132112312321123113112221121113122113111231133221121321132122311211131122211213211321222113222122211211232221123113112221131112311332111213122112311311123112111331121113122112132113121113222112311311221112131221123113112221121113311211131122211211131221131211132221121321132132212321121113121112133221123113112221131112311332111213213211221113122113121113222112132113213221232112111312111213322112132113213221133112132123123112111312211322311211133112111312212221121123222112132113213221133112132123222113223113112221131112311332111213122112311311123112112322211211131221131211132221232112111312111213111213211231132132211211131221131211221321123113213221123113112221131112211322212322211231131122211322111312211312111322211213211321322113311213211331121113122122211211132213211231131122212322211331222113112211"),
    ]
    return test_cases

@pytest.fixture
def roman_to_integer_testcases():
    test_cases = [
        # Test case 1: Simple Roman numeral
        (("I"), 1),  # "I" is 1

        # Test case 2: Basic Roman numeral with addition
        (("III"), 3),  # "III" is 3

        # Test case 3: Roman numeral with subtraction
        (("IV"), 4),  # "IV" is 4 (5 - 1)

        # Test case 4: Another example of addition
        (("VI"), 6),  # "VI" is 6 (5 + 1)

        # Test case 5: Complex Roman numeral
        (("XXI"), 21),  # "XXI" is 21 (10 + 10 + 1)

        # Test case 6: Subtraction with multiple characters
        (("IX"), 9),  # "IX" is 9 (10 - 1)

        # Test case 7: More complex example
        (("LVIII"), 58),  # "LVIII" is 58 (50 + 5 + 1 + 1 + 1)

        # Test case 8: Roman numeral for 99
        (("XCIX"), 99),  # "XCIX" is 99 (100 - 10 + 10 - 1)

        # Test case 9: Large Roman numeral
        (("MCMXCIV"), 1994),  # "MCMXCIV" is 1994 (1000 + 1000 - 100 + 10 - 1)

        # Test case 10: Roman numeral for 400
        (("CD"), 400),  # "CD" is 400 (500 - 100)

        # Test case 11: Roman numeral for 900
        (("CM"), 900),  # "CM" is 900 (1000 - 100)

        # Test case 12: Roman numeral for 58
        (("LVIII"), 58),  # "LVIII" is 58 (50 + 5 + 1 + 1 + 1)

        # Test case 13: Roman numeral for 3999
        (("MMMCMXCIX"), 3999),  # "MMMCMXCIX" is 3999 (3000 + 900 + 90 + 9)

        # Test case 14: Invalid input (empty string)
        (("", 0)),  # Empty string returns 0, no Roman numeral

        # Test case 15: Roman numeral with repeated letters
        (("III"), 3),  # "III" is 3 (1 + 1 + 1)
    ]
    return test_cases

@pytest.fixture
def get_valid_ip_addresses_testcases():
    test_cases = [
        (
            ("25525511135"), 
            ["255.255.11.135", "255.255.111.35"]
        ),
        (
            ("0000"), 
            ["0.0.0.0"]
        ),
        (
            ("1111"), 
            ["1.1.1.1"]
        ),
        (
            ("010010"), 
            ["0.10.0.10","0.100.1.0"]
        ),
        (
            ("1921680111"), 
            ["19.216.80.111","192.168.0.111","192.16.80.111"]
        ),
        (
            ("25505011535"), 
            []
        ),
        (
            ("123456789"), 
            ["123.45.67.89"]
        ),
        (
            ("00000000"), 
            []
        ),
        (
            ("256256256256"), 
            []
        ),
        (
            ("111122223333"), 
            []
        )
    ]
    return test_cases

@pytest.fixture
def snake_string_testcases():
    test_cases = [
        (
            ("Hello World"),
            "e lHloWrdlo"
        ),
        (
            ("SnakeString"),
            "nSnSaetigkr"
        ),
        (
            ("Programming"),
            "ranPormiggm"
        ),
        (
            ("OpenAI"),
            "pIOeAn"
        ),
        (
            ("ExampleTestCase"),
            "xlssEapeetaemTC"
        ),
        (
            ("abcde"),
            "baced"
        ),
        (
            ("abcdefghijk"),
            "bfjacegikdh"
        ),
        (
            (""),
            ""
        ),
        (
            ("a"),
            "a"
        ),
        (
            ("abcdefgh"),
            "bfacegdh"
        )
    ]
    return test_cases

@pytest.fixture
def substring_matching_testcases():
    test_cases = [
        (("hello world", "world"), 6),   # Substring "world" starts at index 6
        (("abcdef", "cd"), 2),           # Substring "cd" starts at index 2
        (("aaaaa", "aa"), 0),            # Substring "aa" first occurs at index 0
        (("abcdabc", "abc"), 0),         # Substring "abc" first occurs at index 0
        (("hello", "lo"), 3),            # Substring "lo" starts at index 3
        (("aaaaaa", "aaa"), 0),          # Substring "aaa" first occurs at index 0
        (("hello world", "o"), 4),       # Substring "o" first occurs at index 4
        (("hello", "z"), -1),            # Substring "z" is not found, so return -1
        (("abcdef", "def"), 3),          # Substring "def" starts at index 3
        (("hello world", " "), 5),       # First space occurs at index 5
        (("test string", " "), 4),       # First space occurs at index 4
        (("find the index", "index"), 9),  # Substring "index" starts at index 8
        (("aaa", "aaaa"), -1),           # Substring "aaaa" not found, so return -1
        (("", "hello"), -1),             # Empty string, so return -1
        (("abcabcabc", "bc"), 1),        # Substring "bc" first occurs at index 1
        (("test", ""), 0),               # Empty substring, index 0 is valid
        (("aaaaa", "bba"), -1),
        (("mississippi", "issip"), 4)
    ]
    return test_cases

def test_palindromic():
    testcases = (
        ("madam", True)          # 'madam' is a palindrome
        , ("racecar", True)      # 'racecar' is a palindrome
        , ("hello", False)       # 'hello' is not a palindrome
        , ("level", True)        # 'level' is a palindrome
        , ("world", False)       # 'world' is not a palindrome
        , ("", True)             # An empty string is considered a palindrome
        , ("a", True)            # A single character is a palindrome
        , ("abccba", True)       # 'abccba' is a palindrome
        , ("abc", False)         # 'abc' is not a palindrome
        , ("noon", True)         # 'noon' is a palindrome
        , ("Was it a car or a cat I saw", False)  # Palindrome with spaces
        , ("hello world", False) # 'hello world' is not a palindrome
        , ("A man a plan a canal Panama", False)  # Palindrome with spaces and case difference
        , ("A man nam A", True) 
    )

    for i, (s, expected) in enumerate(testcases):
        output = is_palindromic(s)
        assert output==expected, f'Failed Testcase {i}: {s=}'

def test_int_to_string():
    COUNTS = 1000
    MIN_VALUE = (2**32-1)*-1
    MAX_VALUE = (2**32-1)

    for _ in range(COUNTS):
        value = randint(MIN_VALUE, MAX_VALUE)
        output = int_to_string(value)
        expected = str(value)
        assert expected == output, f'{value=}, {output=}'

def test_string_to_int():
    COUNTS = 1000
    MIN_VALUE = (2**32-1)*-1
    MAX_VALUE = (2**32-1)

    for _ in range(COUNTS):
        expected = randint(MIN_VALUE, MAX_VALUE)
        input_string = str(expected)
        output = string_to_int(input_string)
        assert expected == output, f'{input_string=}, {output=}'

def test_string_to_int_unrolled():
    COUNTS = 1000
    MIN_VALUE = (2**32-1)*-1
    MAX_VALUE = (2**32-1)

    for _ in range(COUNTS):
        expected = randint(MIN_VALUE, MAX_VALUE)
        input_string = str(expected)
        output = string_to_int_unrolled(input_string)
        assert expected == output, f'{input_string=}, {output=}'

def test_convert_to_base(convert_base_testcases):

    for i, (inputs, expected) in enumerate(convert_base_testcases):
        output = convert_to_base(*inputs)
        assert output == expected, f"Testcase {i+1}: {expected=}, {output=}"

def test_convert_to_base_with_functools(convert_base_testcases):
    for i, (inputs, expected) in enumerate(convert_base_testcases):
        output = convert_to_base_with_functools(*inputs)
        assert output == expected, f"Testcase {i+1}: {inputs=}, {expected=}, {output=}"

def test_spreadsheet_decode_col_id():
    test_cases = [
        # Single letter column names
        (("A",), 1),  # "A" corresponds to 1
        (("B",), 2),  # "B" corresponds to 2
        (("Z",), 26), # "Z" corresponds to 26

        # Two-letter column names
        (("AA",), 27),  # "AA" corresponds to 27
        (("AB",), 28),  # "AB" corresponds to 28
        (("AZ",), 52),  # "AZ" corresponds to 52
        (("BA",), 53),  # "BA" corresponds to 53
        (("BB",), 54),  # "BB" corresponds to 54
        (("ZZ",), 702), # "ZZ" corresponds to 702

        # Three-letter column names
        (("AAA",), 703),  # "AAA" corresponds to 703
        (("AAB",), 704),  # "AAB" corresponds to 704
        (("AAC",), 705),  # "AAC" corresponds to 705
        (("ZZZ",), 18278), # "ZZZ" corresponds to 18278

        # Edge case with a very large column name
        (("XFD",), 16384),  # Largest column name in Excel (Excel's limit is "XFD" which corresponds to 16384)

        # Random cases
        (("C",), 3),    # "C" corresponds to 3
        (("J",), 10),   # "J" corresponds to 10
        (("M",), 13),   # "M" corresponds to 13
        (("AZZ",), 1378),  # Random three-letter column name
    ]
    for i, (inputs, expected) in enumerate(test_cases):
        output = spreadsheet_decode_col_id(*inputs)
        assert output == expected, f"Testcase {i+1}: {inputs=}, {expected=}, {output=}"

def test_replace_and_delete():
    test_cases = [
        # Test case 1: Basic case with multiple 'a' and 'b'
        ((4, ["a", "b", "a", "c", "-"]), (5, ["d", "d", "d", "d", "c"])),
        
        # Test case 2: Only 'a' elements, all 'a' should be replaced with 'd'
        ((3, ["a", "a", "a", "-", "-", "-"]), (6, ["d", "d", "d", "d", "d", "d"])),
        
        # Test case 3: Only 'b' elements, all 'b' should be removed
        ((3, ["b", "b", "b", "-", "-", "-"]), (0, ["-", "-", "-", "-", "-", "-"])),
        
        # Test case 4: No 'a' or 'b', so no changes should happen
        ((3, ["c", "d", "e", "-"]), (3, ["c", "d", "e", "-"])),
        
        # Test case 5: Edge case where size is 0, so no changes should happen
        ((0, ["a", "b", "a", "c", "-", "-", "-"]), (0, ["a", "b", "a", "c", "-", "-", "-"])),
        
        # Test case 6: Mixed case with no 'a' and 'b'
        ((3, ["x", "y", "z", "-"]), (3, ["x", "y", "z", "-"])),
        
        # Test case 7: Mixed array with 'a' and 'b'
        ((6, ["a", "b", "c", "a", "b", "a", "-"]), (7, ["d", "d", "c", "d", "d", "d", "d"])),
        
        # Test case 8: Full array of 'b's (nothing to keep, all 'b' should be removed)
        ((6, ["b", "b", "b", "b", "b", "b"]), (0, ["-", "-", "-", "-", "-", "-"])),
        
        # Test case 9: Mixed array with 'a' and 'b' 2
        ((6, ["a", "b", "a", "c", "a", "b", "-"]), (7, ["d", "d", "d", "d", "c", "d", "d"])),
        
        # Test case 10: Size is larger than the length of the array (will process only available elements)
        ((4, ["a", "c", "a", "b", "-", "-", "-", "-", "-", "-"]), (5, ["d", "d", "c", "d", "d", "-", "-", "-", "-", "-"])),
        
        # Test case 11: All characters are 'a' and we need to transform them
        ((4, ["a", "a", "a", "a", "-", "-", "-", "-"]), (8, ["d", "d", "d", "d", "d", "d", "d", "d"])),
        
        # Test case 12: Single 'a' at the beginning
        ((3, ["a", "b", "c", "-"]), (3, ["d", "d", "c", "-"])),
        
        # Test case 14: Single 'b' in the middle
        ((3, ["a", "b", "a", "-"]), (4, ["d", "d", "d", "d"])),
    ]


    # Our testcase assume '-' as empty char
    for i, (inputs, expected) in enumerate(test_cases):
        size, output = inputs
        expected_size, expected = expected
        original = deepcopy(output)
        output_size = replace_and_delete(size, output)
        output = output[:output_size]
        assert output_size == expected_size and reduce(lambda result, cur: result and cur[0] == cur[1], zip(output, expected), True)\
        , f"Testcase {i+1}: {size=}, {original=}, {expected=}, {output=}"

def test_is_palindrome():
    test_cases = [
        # Test case 1: Simple palindrome, case-insensitive and ignoring spaces
        (("A man, a plan, a canal, Panama", True)),
        
        # Test case 2: Simple palindrome with mixed case
        (("raceCar", True)),
        
        # Test case 3: Non-palindrome string with mixed characters
        (("hello, world!", False)),
        
        # Test case 4: Empty string should be considered a palindrome
        (("", True)),
        
        # Test case 5: Single character string is a palindrome
        (("a", True)),
        
        # Test case 6: String with special characters only should be considered a palindrome
        (("!!", True)),
        
        # Test case 7: Palindrome with digits and mixed characters
        (("12321", True)),
        
        # Test case 8: Non-palindrome string with numbers
        (("12345", False)),
        
        # Test case 9: Case-insensitive palindrome with punctuation
        (("No 'x' in Nixon!", True)),
        
        # Test case 10: Palindrome with spaces and punctuation
        (("Was it a car or a cat I saw?", True)),
        
        # Test case 11: Non-palindrome with mixed alphanumeric characters
        (("This is not a palindrome", False)),
        
        # Test case 12: Palindrome with only non-alphanumeric characters (should be treated as empty)
        (("!!!", True)),
        
        # Test case 13: String that is the same when reversed (with spaces and punctuation)
        (("Madam, in Eden, I'm Adam", True)),
        
        # Test case 14: Non-palindrome with case insensitivity
        (("Able was I ere I saw Elba", True)),
        
        # Test case 15: Non-palindrome with non-alphanumeric characters
        (("hello!!123", False)),

    ]
    for i, (s, expected) in enumerate(test_cases):
        output = is_palindrome(s)
        assert output == expected, f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_reverse_words():
    test_cases = [
        # Test case 1: Simple example with two words
        ((bytearray("Hello World", "utf-8")), (bytearray("World Hello", "utf-8"))),

        # Test case 2: With multiple words
        ((bytearray("The quick brown fox", "utf-8")), (bytearray("fox brown quick The", "utf-8"))),

        # Test case 3: Only one word, no change should happen
        ((bytearray("Python", "utf-8")), (bytearray("Python", "utf-8"))),

        # Test case 4: Empty input (edge case)
        ((bytearray("", "utf-8")), (bytearray("", "utf-8"))),

        # Test case 5: Input with multiple spaces between words
        ((bytearray("Hello   there   world", "utf-8")), (bytearray("world   there   Hello", "utf-8"))),

        # Test case 6: Input with punctuation
        ((bytearray("Hello, world! Python, is great.", "utf-8")), (bytearray("great. is Python, world! Hello,", "utf-8"))),

        # Test case 7: Input with a single word and punctuation
        ((bytearray("Hello,", "utf-8")), (bytearray("Hello,", "utf-8"))),
    ]
    for i, (s, expected) in enumerate(test_cases):
        output = reverse_words(s)
        assert reduce(lambda result, value: result and value[0]==value[1], zip(output, expected), True), f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_phone_mnemonic():
    test_cases = [
        # Test case 1: Simple case with two digits
        (("23"), ['AD', 'AE', 'AF', 'BD', 'BE', 'BF', 'CD', 'CE', 'CF']),

        # Test case 2: Case with three digits (ABC, DEF, GHI)
        (("234"), ['ADG', 'ADH', 'ADI', 'AEG', 'AEH', 'AEI', 'AFG', 'AFH', 'AFI', 
                    'BDG', 'BDH', 'BDI', 'BEG', 'BEH', 'BEI', 'BFG', 'BFH', 'BFI', 
                    'CDG', 'CDH', 'CDI', 'CEG', 'CEH', 'CEI', 'CFG', 'CFH', 'CFI']),

        # Test case 3: Single digit
        (("2"), ['A', 'B', 'C']),

        # Test case 4: Edge case with all 0s (Should return only '0' since it maps to itself)
        (("0000000"), ['0000000']),

        # Test case 5: Larger case with multiple digits
        (("2345"), ['ADGJ', 'ADGK', 'ADGL', 'ADHJ', 'ADHK', 'ADHL', 'ADIJ', 'ADIK',
                    'ADIL', 'AEGJ', 'AEGK', 'AEGL', 'AEHJ', 'AEHK', 'AEHL', 'AEIJ',
                    'AEIK', 'AEIL', 'AFGJ', 'AFGK', 'AFGL', 'AFHJ', 'AFHK', 'AFHL',
                    'AFIJ', 'AFIK', 'AFIL', 'BDGJ', 'BDGK', 'BDGL', 'BDHJ', 'BDHK', 
                    'BDHL', 'BDIJ', 'BDIK', 'BDIL', 'BEGJ', 'BEGK', 'BEGL', 'BEHJ', 
                    'BEHK', 'BEHL', 'BEIJ', 'BEIK', 'BEIL', 'BFGJ', 'BFGK', 'BFGL', 
                    'BFHJ', 'BFHK', 'BFHL', 'BFIJ', 'BFIK', 'BFIL', 'CDGJ', 'CDGK', 
                    'CDGL', 'CDHJ', 'CDHK', 'CDHL', 'CDIJ', 'CDIK', 'CDIL', 'CEGJ', 
                    'CEGK', 'CEGL', 'CEHJ', 'CEHK', 'CEHL', 'CEIJ', 'CEIK', 'CEIL', 
                    'CFGJ', 'CFGK', 'CFGL', 'CFHJ', 'CFHK', 'CFHL', 'CFIJ', 'CFIK', 'CFIL']),
    ]

    for i, (s, expected) in enumerate(test_cases):
        output = phone_mnemonic(s)
        output = list(sorted(set(output)))
        expected = list(sorted(set(expected)))
        assert len(output) == len(expected) and reduce(lambda result, value: result and value[0]==value[1], zip(output, expected), True), f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_look_and_say(look_and_say_testcases):
    for i, (s, expected) in enumerate(look_and_say_testcases):
        output = look_and_say(s)
        assert output == expected, f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_look_and_say_pythonic(look_and_say_testcases):
    for i, (s, expected) in enumerate(look_and_say_testcases):
        output = look_and_say_pythonic(s)
        assert output == expected, f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_roman_to_integer(roman_to_integer_testcases):
    for i, (s, expected) in enumerate(roman_to_integer_testcases):
        output = roman_to_integer(s)
        assert output == expected, f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_roman_to_integer_pythonic(roman_to_integer_testcases):
    for i, (s, expected) in enumerate(roman_to_integer_testcases):
        output = roman_to_integer_pythonic(s)
        assert output == expected, f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_get_valid_ip_addresses(get_valid_ip_addresses_testcases):
    for i, (s, expected) in enumerate(get_valid_ip_addresses_testcases):
        output = get_valid_ip_addresses(s)
        output = list(sorted(set(output)))
        expected = list(sorted(set(expected)))
        assert len(output) == len(expected) and reduce(lambda res, cur: res and (cur[0] == cur[1]), zip(output, expected), True), f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_get_valid_ip_addresses_iterative(get_valid_ip_addresses_testcases):
    for i, (s, expected) in enumerate(get_valid_ip_addresses_testcases):
        output = get_valid_ip_addresses_iterative(s)
        output = list(sorted(set(output)))
        expected = list(sorted(set(expected)))
        assert len(output) == len(expected) and reduce(lambda res, cur: res and (cur[0] == cur[1]), zip(output, expected), True), f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_snake_string(snake_string_testcases):
    for i, (s, expected) in enumerate(snake_string_testcases):
        output = snake_string(s)
        assert output == expected, f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_snake_string_pythonic(snake_string_testcases):
    for i, (s, expected) in enumerate(snake_string_testcases):
        output = snake_string_pythonic(s)
        assert output == expected, f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_run_length_encoding():
    test_cases = [
        (("aaabbbccc"), "3a3b3c"), 
        (("abcdef"), "1a1b1c1d1e1f"), 
        (("a"), "1a"), 
        (("aabbcccddddeee"), "2a2b3c4d3e"), 
        (("zzzzzz"), "6z"), 
        (("xyz"), "1x1y1z"), 
        (("", "")), 
        (("aAaaBB"), "1a1A2a2B"), 
        (("$$$$%%%^^^"), "4$3%3^")
    ]

    for i, (s, expected) in enumerate(test_cases):
        encoded = encode_run_length_encoding(s)
        assert encoded == expected, f'Testcase {i+1}: {s=}, {encoded=}, {expected=}'

        decoded = decode_run_length_encoding(encoded)
        assert s == decoded, f'Testcase {i+1}: {s=}, {decoded=}, {s=}'

def test_rabin_karp_naive(substring_matching_testcases):
    for i, (s, expected) in enumerate(substring_matching_testcases):
        output = rabin_karp_naive(*s)
        assert output == expected, f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_rabin_karp(substring_matching_testcases):
    for i, (s, expected) in enumerate(substring_matching_testcases):
        output = rabin_karp(*s)
        assert output == expected, f'Testcase {i+1}: {s=}, {output=}, {expected=}'

def test_kmp(substring_matching_testcases):
    for i, (s, expected) in enumerate(substring_matching_testcases):
        output = kmp(*s)
        assert output == expected, f'Testcase {i+1}: {s=}, {output=}, {expected=}'