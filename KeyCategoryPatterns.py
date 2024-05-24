KeyCategoryPatterns = [
    {
        "regEx": r"\b(?:purchase\s|asking\s)?(?:price|asking|pp|purchase).{0,4}\$?\d(?:[\d,])+k?\b",
        "description": "matches (purchase or asking)? (price or asking or pp)",
        "for": "SFH Deal or Land Deal"
    },
    {
        "regEx": r"\bflips?\b",
        "description": "matches words flip or flips",
        "for": "SFH Deal"
    },
    {
        "regEx": r"#(?:homes?forsale|forsale)\b",
        "description": "matches words flip or flips",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\bproperty\sin\b",
        "description": "matches words flip or flips",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\bfoundation\srepair\b",
        "description": "matches words flip or flips",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\benjoy\s(?:[\w\s]){0,20}home\b",
        "description": "matches words flip or flips",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\b\d[-\/]\d(?:[-\/]\d)?\b",
        "description": "matches digit[/-]digit([/-]digit) for bed bath etc...",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\d.{0,4}(?:bed(?:room)?|bd|br)s?\b|\d.{0,4}(?:bath(?:room)?|ba|bt)s?\b",
        "description": "matches digit[/-]digit([/-]digit) for bed bath etc...",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\barv|after[- ]retail[- ]value\b",
        "description": "matches arv or after repair value",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\btenant|renter\b",
        "description": "matches cash flowing property",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\bprovide.+list",
        "description": "matches cash flowing property",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\bbuyers?\slist",
        "description": "matches cash flowing property",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\boff.market.+(?:property|house|home|(?:du|tri|quad)plex)\b",
        "description": "matches cash flowing property",
        "for": "SFH Deal"
    },
    {
        "regEx": r"\b(?:vacant|available)\s(?:lot|land)|(?:lot|land)\savailable\b",
        "description": "matches '(vacant or available) (lot or land) or (lot or land) available'",
        "for": "Land Deal"
    },
    {
        "regEx": r"\b(?:residential|commercial)\s(?:lot|land|acres?)\b",
        "description": "matches '(residential or commercial) (lot or land or acres(s))'",
        "for": "Land Deal"
    },
    {
        "regEx": r"\bbuild\sready\slots?\b",
        "description": "matches build ready lot(s)",
        "for": "Land Deal"
    },
    {
        "regEx": r"\btear.?down\b",
        "description": "matches tear( )down",
        "for": "Land Deal"
    },
        {
        "regEx": r"\bzoned|zoning\b",
        "description": "matches zoned or zoning",
        "for": "Land Deal"
    },
    {
        "regEx": r"\bbuild(?:er)?\sopportunity\b",
        "description": "matches build(er) opportunity",
        "for": "Land Deal"
    },
    {
        "regEx": r"\b(?:acre|sqft)\slots?\sin\b",
        "description": "matches (sqft or acre) lot(s)",
        "for": "Land Deal"
    },
    {
        "regEx": r"\b(?:\d+(?:\+|[+-]\/[+-])?|an).acres?\s(?:for\ssale|in|near|of\sland)\b",
        "description": "matches (number(+) or an) acre(s)",
        "for": "Land Deal"
    },
    {
        "regEx": r"\bflood\szone|utilities|sewer\b",
        "description": "matches flood zone or utilities or sewer",
        "for": "Land Deal"
    },
    {
        "regEx": r"\b(?:land|lot)\s(?:size|for\ssale)\b",
        "description": "matches (land or lot) (size or for sale)",
        "for": "Land Deal"
    },
    {
        "regEx": r"\boff.market\s(?:land|lots?)\b",
        "description": "matches off(any character)market (any amount of words) lot(s)",
        "for": "Land Deal"
    },
    {
        "regEx": r"\bnew\sbuilds?\sarea\b",
        "description": "matches new build(s) area",
        "for": "Land Deal"
    },
    {
        "regEx": r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+|multi.use)\slots?\b",
        "description": "matches (number or multi.use) lots",
        "for": "Land Deal"
    },
    {
        "regEx": r"\bempty\slots?\b",
        "description": "matches empty lot(s)",
        "for": "Land Deal"
    },
    {
        "regEx": r"\b(?:0|zero)\s(?:beds|baths|bathrooms|bedrooms)\b",
        "description": "matches (0 or zero) (beds or baths or bedrooms or bathrooms)",
        "for": "Land Deal"
    },
    {
        "regEx": r"\bnew\sbuilds?\sarv\b",
        "description": "matches new build(s) arv",
        "for": "Land Deal"
    },
    {
        "regEx": r"(?:^|i'?m\s|i\sam\s|we\sare\s|we'?re\s)looking\b",
        "description": "matches im looking for",
        "for": "None"
    },
    {
        "regEx": r"\bloans?|lender\b",
        "description": "matches loan(s) or lender",
        "for": "None"
    },
    {
        "regEx": r"\bmfh|multi.family\b",
        "description": "matches mfh or multi family",
        "for": "None"
    },
    {
        "regEx": r"\bapartment\b",
        "description": "matches apartment",
        "for": "None"
    },
    {
        "regEx": r"\bparking.(?:site|lot)\b",
        "description": "matches parking (site or lot)",
        "for": "None"
    },
    {
        "regEx": r"\badjacent\s(?:lot|land)\b",
        "description": "matches adjacent (lot or land)",
        "for": "Land Deal"
    }
]