from enum import Enum
import re


def _to_inch(n, dpi=300) -> float:
    return round(n / dpi, 1)


def PixToInch(width, height, dpi):
    return (_to_inch(width, dpi), _to_inch(height, dpi))


class FigSize(Enum):
    _dpi = 300
    DPI = _dpi
    XXS1_1 = PixToInch(500, 500, _dpi)
    XS1_1 = PixToInch(1000, 1000, _dpi)
    S1_1 = PixToInch(1500, 1500, _dpi)
    M1_1 = PixToInch(2000, 2000, _dpi)
    L1_1 = PixToInch(2500, 2500, _dpi)
    XL1_1 = PixToInch(3000, 3000, _dpi)
    XXL1_1 = PixToInch(5000, 5000, _dpi)
    XXXL1_1 = PixToInch(10000, 10000, _dpi)
    ENORMOUS1_1 = PixToInch(15000, 15000, _dpi)
    XE1_1 = PixToInch(50000, 50000, _dpi)
    XXS16_9 = (1.7, 0.9)  #  500 × 281
    XS16_9 = (3.3, 1.9)  # 1000 × 562
    S16_9 = (5.0, 2.8)  # 1500 × 844
    M16_9 = (6.7, 3.4)  # 2000 × 1125
    L16_9 = (8.3, 4.2)  # 2500 × 1406
    XL16_9 = (10.0, 5.0)  # 3000 × 1688
    XXL16_9 = (11.7, 5.9)  # 3500 × 1969
    XXXL16_9 = (13.3, 6.8)  # 4000 × 2250
    ENORMOUS16_9 = (16.7, 9.4)  # 5000 × 2812
    XE16_9 = (33.3, 18.8)  # 10000 × 5625
    XXS4_3 = (1.7, 1.3)  #  500 × 375
    XS4_3 = (3.3, 2.5)  # 1000 × 750
    S4_3 = (5.0, 3.8)  # 1500 × 1125
    M4_3 = (6.7, 5.0)  # 2000 × 1500
    L4_3 = (8.3, 6.3)  # 2500 × 1875
    XL4_3 = (10.0, 7.5)  # 3000 × 2250
    XXL4_3 = (11.7, 8.8)  # 3500 × 2625
    XXXL4_3 = (13.3, 10.0)  # 4000 × 3000
    ENORMOUS4_3 = (16.7, 12.5)  # 5000 × 3750
    XE4_3 = (33.3, 25.0)  # 10000 × 7500


class preproc:
    @staticmethod
    def normalize_email(email_str):
        if not email_str:
            return None
        match = re.search(r"[\w\.-]+@[\w\.-]+", email_str)
        if match:
            return match.group(0).lower()
        return None

    @staticmethod
    def extract_journal_ref(text: str):
        """
        Extract Journal-ref from Abstract text
        """
        journalref = re.search(
            r"^\s*Journal[- ]ref\s*:\s*(.+)", text, re.I | re.MULTILINE
        )
        if journalref:
            return journalref.group(1).strip()
        return None

    @staticmethod
    def extract_pages(comment_str: str):
        """
        Extract page number from an abstract comment string.
        Returns an int or None.
        """
        match = re.search(
            r"(?i)(\b(?:p{1,2}|pages?)\.?\s*(\d+)\b|\b(\d+)\s*(?:p{1,2}|pages?)\.?\b)",
            comment_str,
        )
        if match:
            # groups: match.group(2) OR match.group(3) contains the number
            return int(match.group(2) or match.group(3))
        return None

    @staticmethod
    def extract_date_fields(text):
        """
        Extract pubblication dates and revision
        """
        date_publish = None
        date_revised = None

        # Date originale
        match = re.search(r"^Date:\s*(.+?)(?:\s+\(\d+kb\))?$", text, re.MULTILINE)
        if match:
            date_publish = match.group(1).strip()

        # Date revised (es. 'Date (revised v2): ...')
        match_rev = re.search(
            r"^Date\s*\(revised.*?\):\s*(.+?)(?:\s+\(\d+kb\))?$", text, re.MULTILINE
        )
        if match_rev:
            date_revised = match_rev.group(1).strip()

        return date_publish, date_revised

    @staticmethod
    def extract_fields(text: str) -> dict:
        """
        Extract fields from abstract
        """

        data: dict[str, object] = {
            "email": None,
        }

        keys = data.keys()

        for line in text.splitlines():
            line = line.lower().strip()
            tag, _, content = line.partition(":")
            if tag == "from":
                data["email"] = preproc.normalize_email(content.strip())

        return data

    @staticmethod
    def extract_domain(email):
        if not isinstance(email, str):
            return None
        if "@" not in email:
            return None

        domain = email.split("@", 1)[1].lower()

        # LIKE-style match contro all_universities.csv
        cond_a = universities["domains"].str.contains(domain, case=False, na=False)
        cond_b = universities["domains"].apply(
            lambda d: isinstance(d, str) and d.lower() in domain
        )
        mask = cond_a | cond_b

        # restituisco SOLO name e country
        uni_match = universities.loc[mask, ["name", "alpha_two_code"]]

        if not uni_match.empty:
            row = uni_match.iloc[0]
            return {"name": row["name"], "country": row["alpha_two_code"]}

        # FALLBACK: ROR via TLD2
        m = re.search(r"([a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)$", domain)
        if m:
            tld2 = m.group(1)
            ror_match = ror.loc[ror["tld2"].eq(tld2), ["name", "country.country_code"]]

            if not ror_match.empty:
                row = ror_match.iloc[0]
                return {"name": row["name"], "country": row["country.country_code"]}

        return None
