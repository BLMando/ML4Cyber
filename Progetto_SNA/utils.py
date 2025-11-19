from enum import Enum
import re


def _to_inch(n, dpi=300):
    return n / dpi


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
    XXL1_1 = PixToInch(3500, 3500, _dpi)
    XXXL1_1 = PixToInch(4000, 4000, _dpi)
    ENORMOUS1_1 = PixToInch(5000, 5000, _dpi)
    XE1_1 = PixToInch(10000, 10000, _dpi)


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
