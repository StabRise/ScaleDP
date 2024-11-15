
LANGUAGE_TO_TESSERACT_CODE = {
    'Afrikaans': 'afr',
    'Amharic': 'amh',
    'Arabic': 'ara',
    'Assamese': 'asm',
    'Azerbaijani': 'aze',
    'Belarusian': 'bel',
    'Bulgarian': 'bul',
    'Bengali': 'ben',
    'Breton': 'bre',
    'Bosnian': 'bos',
    'Catalan': 'cat',
    'Czech': 'ces',
    'Welsh': 'cym',
    'Danish': 'dan',
    'German': 'deu',
    'Greek': 'ell',
    'English': 'eng',
    'Esperanto': 'epo',
    'Spanish': 'spa',
    'Estonian': 'est',
    'Basque': 'eus',
    'Persian': 'fas',
    'Finnish': 'fin',
    'French': 'fra',
    'Western Frisian': 'fry',
    'Irish': 'gle',
    'Scottish Gaelic': 'gla',
    'Galician': 'glg',
    'Gujarati': 'guj',
    'Hausa': 'hau',
    'Hebrew': 'heb',
    'Hindi': 'hin',
    'Croatian': 'hrv',
    'Hungarian': 'hun',
    'Armenian': 'hye',
    'Indonesian': 'ind',
    'Icelandic': 'isl',
    'Italian': 'ita',
    'Japanese': 'jpn',
    'Javanese': 'jav',
    'Georgian': 'kat',
    'Kazakh': 'kaz',
    'Khmer': 'khm',
    'Kannada': 'kan',
    'Korean': 'kor',
    'Kurdish': 'kur',
    'Kyrgyz': 'kir',
    'Latin': 'lat',
    'Lao': 'lao',
    'Lithuanian': 'lit',
    'Latvian': 'lav',
    'Malagasy': 'mlg',
    'Macedonian': 'mkd',
    'Malayalam': 'mal',
    'Mongolian': 'mon',
    'Marathi': 'mar',
    'Malay': 'msa',
    'Burmese': 'mya',
    'Nepali': 'nep',
    'Dutch': 'nld',
    'Norwegian': 'nor',
    'Oromo': 'orm',
    'Oriya': 'ori',
    'Punjabi': 'pan',
    'Polish': 'pol',
    'Pashto': 'pus',
    'Portuguese': 'por',
    'Romanian': 'ron',
    'Russian': 'rus',
    'Sanskrit': 'san',
    'Sindhi': 'snd',
    'Sinhala': 'sin',
    'Slovak': 'slk',
    'Slovenian': 'slv',
    'Somali': 'som',
    'Albanian': 'sqi',
    'Serbian': 'srp',
    'Sundanese': 'sun',
    'Swedish': 'swe',
    'Swahili': 'swa',
    'Tamil': 'tam',
    'Telugu': 'tel',
    'Thai': 'tha',
    'Tagalog': 'tgl',
    'Turkish': 'tur',
    'Uyghur': 'uig',
    'Ukrainian': 'ukr',
    'Urdu': 'urd',
    'Uzbek': 'uzb',
    'Vietnamese': 'vie',
    'Xhosa': 'xho',
    'Yiddish': 'yid',
    'Chinese': 'chi_sim',
}

TESSERACT_CODE_TO_LANGUAGE = {v:k for k,v in LANGUAGE_TO_TESSERACT_CODE.items()}

CODE_TO_LANGUAGE = {
    "_math": "Math",
    'af': 'Afrikaans',
    'am': 'Amharic',
    'ar': 'Arabic',
    'as': 'Assamese',
    'az': 'Azerbaijani',
    'be': 'Belarusian',
    'bg': 'Bulgarian',
    'bn': 'Bengali',
    'br': 'Breton',
    'bs': 'Bosnian',
    'ca': 'Catalan',
    'cs': 'Czech',
    'cy': 'Welsh',
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'eo': 'Esperanto',
    'es': 'Spanish',
    'et': 'Estonian',
    'eu': 'Basque',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fr': 'French',
    'fy': 'Western Frisian',
    'ga': 'Irish',
    'gd': 'Scottish Gaelic',
    'gl': 'Galician',
    'gu': 'Gujarati',
    'ha': 'Hausa',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'hu': 'Hungarian',
    'hy': 'Armenian',
    'id': 'Indonesian',
    'is': 'Icelandic',
    'it': 'Italian',
    'ja': 'Japanese',
    'jv': 'Javanese',
    'ka': 'Georgian',
    'kk': 'Kazakh',
    'km': 'Khmer',
    'kn': 'Kannada',
    'ko': 'Korean',
    'ku': 'Kurdish',
    'ky': 'Kyrgyz',
    'la': 'Latin',
    'lo': 'Lao',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'mg': 'Malagasy',
    'mk': 'Macedonian',
    'ml': 'Malayalam',
    'mn': 'Mongolian',
    'mr': 'Marathi',
    'ms': 'Malay',
    'my': 'Burmese',
    'ne': 'Nepali',
    'nl': 'Dutch',
    'no': 'Norwegian',
    'om': 'Oromo',
    'or': 'Oriya',
    'pa': 'Punjabi',
    'pl': 'Polish',
    'ps': 'Pashto',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sa': 'Sanskrit',
    'sd': 'Sindhi',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'so': 'Somali',
    'sq': 'Albanian',
    'sr': 'Serbian',
    'su': 'Sundanese',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'tl': 'Tagalog',
    'tr': 'Turkish',
    'ug': 'Uyghur',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'uz': 'Uzbek',
    'vi': 'Vietnamese',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'zh': 'Chinese',
}

LANGUAGE_TO_CODE = {v: k for k, v in CODE_TO_LANGUAGE.items()}