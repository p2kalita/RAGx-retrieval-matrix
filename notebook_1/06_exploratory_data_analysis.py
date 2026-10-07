"""
Section 6: Exploratory Data Analysis

Exploratory Data Analysis (EDA) is conducted to understand the structural and linguistic characteristics of the cleaned legal corpus before chunk generation and embedding creation. The objective of this analysis is not merely descriptive but to provide empirical evidence supporting the design decisions adopted in the NYAYaRAG retrieval pipeline.

The analysis examines corpus size, judgment length distribution, sentence and paragraph statistics, lexical characteristics, and the distribution of frequently occurring legal terminology and statutes. These characteristics help explain the variability of Supreme Court judgments and justify the need for text chunking, dense semantic embeddings, and hybrid retrieval strategies.

The findings from this section establish a comprehensive understanding of the corpus and serve as the basis for selecting appropriate preprocessing techniques, chunking parameters, and retrieval models in the subsequent stages of the framework.
"""

from collections import Counter
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd


def compute_corpus_statistics(corpus_df):
    """
    6.1 Corpus Statistics

    This section summarises the overall size and characteristics of the legal corpus. The statistics provide a high-level overview of the number of judgments and the volume of textual data available for retrieval.

    Understanding the scale of the corpus is important because it influences subsequent preprocessing decisions, including chunk generation, embedding computation, vector indexing, and retrieval efficiency.

    Observation:
    The cleaned corpus comprises 782 Supreme Court judgments, containing 6,820,528 words and 40,992,347 characters. On average, each judgment contains 8,721.90 words (median: 5,240 words) and 52,419.88 characters (median: 31,398 characters).

    Judgment lengths vary substantially, ranging from 320 to 144,369 words, indicating considerable heterogeneity in the corpus. This variability demonstrates that Supreme Court judgments differ markedly in complexity and size, making fixed-length document retrieval inefficient.

    These descriptive statistics justify the adoption of a chunk-based retrieval strategy in the NYAYaRAG framework, enabling long judgments to be partitioned into manageable semantic units while preserving retrieval effectiveness.
    """
    total_judgments = len(corpus_df)

    total_characters = corpus_df["clean_characters"].sum()
    total_words = corpus_df["clean_words"].sum()

    mean_words = corpus_df["clean_words"].mean()
    median_words = corpus_df["clean_words"].median()

    min_words = corpus_df["clean_words"].min()
    max_words = corpus_df["clean_words"].max()

    mean_characters = corpus_df["clean_characters"].mean()
    median_characters = corpus_df["clean_characters"].median()

    print("=" * 70)
    print("Corpus Statistics")
    print("=" * 70)

    print(f"Total Judgments          : {total_judgments:,}")
    print(f"Total Words              : {total_words:,}")
    print(f"Total Characters         : {total_characters:,}")

    print()

    print(f"Average Words/Judgment   : {mean_words:,.2f}")
    print(f"Median Words/Judgment    : {median_words:,.0f}")
    print(f"Minimum Words            : {min_words:,}")
    print(f"Maximum Words            : {max_words:,}")

    print()

    print(f"Average Characters       : {mean_characters:,.2f}")
    print(f"Median Characters        : {median_characters:,.0f}")


def plot_judgment_length_distribution(corpus_df):
    """
    6.2 Judgment Length Distribution

    Supreme Court judgments differ considerably in length because they address a wide range of legal issues, procedural histories, and judicial reasoning. Examining the distribution of judgment lengths provides insight into the variability of the corpus and helps justify the need for chunk-based retrieval.

    The distribution is analysed using the number of words in each cleaned judgment. Since the retrieval pipeline operates on the cleaned corpus, all subsequent analyses are based on the cleaned text.

    Observation:
    The histogram demonstrates that the distribution of judgment lengths is highly positively skewed. Most Supreme Court judgments contain fewer than approximately 15,000 words, while only a small number of judgments exceed this range and extend to more than 140,000 words.

    The median judgment length (5,240 words) is substantially lower than the mean (8,722 words), indicating that a relatively small number of exceptionally long judgments increase the overall average. This confirms the presence of a long right tail in the distribution.

    The observed variability in judgment length demonstrates that Supreme Court decisions differ considerably in complexity and volume of legal reasoning. Consequently, indexing complete judgments as single retrieval units would be inefficient, particularly for dense embedding models with input length limitations. These findings provide empirical support for adopting a chunk-based indexing strategy in the NYAYaRAG framework.
    """
    plt.figure(figsize=(10, 6), dpi=150)

    plt.hist(
        corpus_df["clean_words"],
        bins=40,
        edgecolor="black",
        alpha=0.8
    )

    # Mean
    plt.axvline(
        corpus_df["clean_words"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {corpus_df['clean_words'].mean():.0f}"
    )

    # Median
    plt.axvline(
        corpus_df["clean_words"].median(),
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Median = {corpus_df['clean_words'].median():.0f}"
    )

    plt.title("Distribution of Judgment Lengths (Word Count)", fontsize=14)
    plt.xlabel("Number of Words", fontsize=12)
    plt.ylabel("Number of Judgments", fontsize=12)

    plt.grid(axis="y", alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()


def analyze_judgment_length_outliers(corpus_df):
    """
    6.3 Judgment Length Outlier Analysis

    While the histogram illustrates the overall distribution of judgment lengths, a box plot provides a concise visual summary of the median, interquartile range, and potential outliers. Outlier detection is particularly important because extremely long judgments can disproportionately influence corpus statistics and significantly increase the number of text chunks generated during indexing.

    Outliers are identified using the Interquartile Range (IQR) method, which classifies observations beyond 1.5 × IQR above the third quartile or below the first quartile as potential outliers.

    Observation:
    The box plot confirms that the distribution of judgment lengths is highly positively skewed and contains numerous upper outliers. The median judgment length is 5,240.50 words, while the upper quartile is 8,802.50 words, indicating that 75% of the judgments contain fewer than approximately 8,800 words.

    Using the Interquartile Range (IQR) method, the upper outlier threshold was calculated as 17,244.88 words. A total of 73 judgments (9.34% of the corpus) exceed this threshold and are therefore classified as unusually long documents. No lower outliers were identified because the calculated lower bound is negative and judgment lengths cannot assume negative values.

    The presence of these exceptionally long judgments highlights the substantial variability in document size within the corpus. Such documents would produce disproportionately large embeddings if indexed as complete texts and could adversely affect retrieval efficiency. These findings further justify the adoption of chunk-based indexing in the NYAYaRAG framework, allowing lengthy judgments to be partitioned into manageable semantic units while preserving retrieval quality.
    """
    plt.figure(figsize=(10, 2.5), dpi=150)

    plt.boxplot(
        corpus_df["clean_words"],
        vert=False,
        patch_artist=True
    )

    plt.title("Box Plot of Judgment Lengths")
    plt.xlabel("Number of Words")

    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    plt.show()

    # IQR-Based Outlier Detection
    Q1 = corpus_df["clean_words"].quantile(0.25)
    Q3 = corpus_df["clean_words"].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    outliers = corpus_df[
        (corpus_df["clean_words"] < lower_bound) |
        (corpus_df["clean_words"] > upper_bound)
    ]

    print("=" * 70)
    print("Outlier Analysis")
    print("=" * 70)

    print(f"Q1                  : {Q1:,.2f}")
    print(f"Median              : {corpus_df['clean_words'].median():,.2f}")
    print(f"Q3                  : {Q3:,.2f}")
    print(f"IQR                 : {IQR:,.2f}")

    print()

    print(f"Lower Bound         : {lower_bound:,.2f}")
    print(f"Upper Bound         : {upper_bound:,.2f}")

    print()

    print(f"Number of Outliers  : {len(outliers):,}")
    print(f"Percentage          : {(len(outliers)/len(corpus_df))*100:.2f}%")

    # Largest Outlier Judgments
    print("\nLargest Outlier Judgments:\n")
    print(
        outliers.sort_values(
            "clean_words",
            ascending=False
        )[
            [
                "title",
                "citation",
                "clean_words"
            ]
        ].head(10)
    )

    # Top 10 Longest Judgments
    longest_judgments = (
        corpus_df[
            [
                "title",
                "citation",
                "decision_date",
                "clean_words",
                "clean_characters"
            ]
        ]
        .sort_values("clean_words", ascending=False)
        .head(10)
    )

    print("=" * 70)
    print("Top 10 Longest Judgments")
    print("=" * 70)
    print(longest_judgments)

    # Top 10 Shortest Judgments
    shortest_judgments = (
        corpus_df[
            [
                "title",
                "citation",
                "decision_date",
                "clean_words",
                "clean_characters"
            ]
        ]
        .sort_values("clean_words", ascending=True)
        .head(10)
    )

    print("=" * 70)
    print("Top 10 Shortest Judgments")
    print("=" * 70)
    print(shortest_judgments)

    return outliers


def compute_and_plot_sentence_statistics(corpus_df):
    """
    7.4 Sentence Statistics

    The number of sentences in a judgment provides an additional measure of document complexity beyond word count alone. While two judgments may contain a similar number of words, they may differ substantially in sentence structure, level of detail, and organisation of legal reasoning.

    Analysing sentence counts helps characterise the internal structure of the corpus and provides useful evidence for selecting an appropriate chunking strategy. Judgments containing a large number of sentences are more likely to require multiple semantically coherent chunks during indexing, whereas shorter judgments may be represented using a single chunk. Consequently, sentence-level statistics complement the document-length analysis by offering further insight into the variability of the legal corpus.

    Observation:
    The cleaned corpus contains a total of 342,405 sentences, with an average of 437.86 sentences and a median of 273 sentences per judgment. Sentence counts range from 35 to 7,846, indicating substantial variation in the internal structure and complexity of Supreme Court judgments.

    The difference between the mean and median suggests a positively skewed distribution, where a relatively small number of exceptionally long judgments contain a disproportionately large number of sentences. This pattern is consistent with the document-length analysis and reflects the presence of constitutionally significant or legally complex decisions requiring extensive judicial reasoning.

    The wide variation in sentence counts further supports the need for chunk-based indexing. Representing lengthy judgments as multiple semantically coherent chunks enables more efficient embedding generation and retrieval while preserving the contextual integrity of legal reasoning.
    """
    # Count sentences
    corpus_df["sentence_count"] = corpus_df["clean_text"].apply(
        lambda text: len(
            [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        )
    )

    print("=" * 70)
    print("Sentence Statistics")
    print("=" * 70)

    print(f"Total Sentences           : {corpus_df['sentence_count'].sum():,}")
    print(f"Average Sentences         : {corpus_df['sentence_count'].mean():.2f}")
    print(f"Median Sentences          : {corpus_df['sentence_count'].median():.0f}")
    print(f"Minimum Sentences         : {corpus_df['sentence_count'].min():,}")
    print(f"Maximum Sentences         : {corpus_df['sentence_count'].max():,}")

    print("\nSummary Statistics\n")
    print(
        corpus_df["sentence_count"]
        .describe()
        .round(2)
        .to_frame(name="Value")
    )

    plt.figure(figsize=(10, 6), dpi=150)

    plt.hist(
        corpus_df["sentence_count"],
        bins=40,
        edgecolor="black",
        alpha=0.8
    )

    plt.axvline(
        corpus_df["sentence_count"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {corpus_df['sentence_count'].mean():.0f}"
    )

    plt.axvline(
        corpus_df["sentence_count"].median(),
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Median = {corpus_df['sentence_count'].median():.0f}"
    )

    plt.title("Distribution of Sentence Counts")
    plt.xlabel("Number of Sentences")
    plt.ylabel("Number of Judgments")

    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.show()
    return corpus_df


def compute_and_plot_paragraph_statistics(corpus_df):
    """
    7.5 Paragraph Statistics

    Paragraphs represent natural semantic boundaries within judicial decisions, with each paragraph typically addressing a specific legal issue, factual finding, statutory provision, or judicial argument. Analysing paragraph counts provides insight into the structural organisation of Supreme Court judgments and complements the earlier analyses of document and sentence lengths.

    Understanding paragraph-level characteristics is particularly important for retrieval-augmented generation because paragraph boundaries can serve as meaningful chunking units. Examining the distribution of paragraphs therefore helps assess the structural complexity of the corpus and supports the selection of an effective chunking strategy for the NYAYaRAG framework.

    Observation:
    The cleaned corpus contains 47,258 paragraphs, with an average of 60.43 paragraphs and a median of 37 paragraphs per judgment. Paragraph counts range from 6 to 1,773, indicating substantial variation in the structural organisation of Supreme Court decisions.

    The difference between the mean and median indicates a positively skewed distribution, where a relatively small number of lengthy judgments contain exceptionally large numbers of paragraphs. This observation is consistent with the earlier analyses of document lengths and sentence counts, confirming that judicial decisions vary considerably in both size and structural complexity.

    From a retrieval perspective, paragraphs represent meaningful semantic units because individual paragraphs typically discuss a single legal proposition, statutory interpretation, factual finding, or judicial conclusion. The observed paragraph structure therefore provides empirical support for preserving semantic coherence during chunk generation, thereby improving the quality of embedding representations and downstream retrieval within the NYAYaRAG framework.
    """
    # Count paragraphs (split on blank lines)
    corpus_df["paragraph_count"] = corpus_df["clean_text"].apply(
        lambda text: len(
            [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        )
    )

    print("=" * 70)
    print("Paragraph Statistics")
    print("=" * 70)

    print(f"Total Paragraphs         : {corpus_df['paragraph_count'].sum():,}")
    print(f"Average Paragraphs       : {corpus_df['paragraph_count'].mean():.2f}")
    print(f"Median Paragraphs        : {corpus_df['paragraph_count'].median():.0f}")
    print(f"Minimum Paragraphs       : {corpus_df['paragraph_count'].min():,}")
    print(f"Maximum Paragraphs       : {corpus_df['paragraph_count'].max():,}")

    print("\nSummary Statistics\n")
    print(
        corpus_df["paragraph_count"]
        .describe()
        .round(2)
        .to_frame(name="Value")
    )

    plt.figure(figsize=(10, 6), dpi=150)

    plt.hist(
        corpus_df["paragraph_count"],
        bins=40,
        edgecolor="black",
        alpha=0.8
    )

    plt.axvline(
        corpus_df["paragraph_count"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {corpus_df['paragraph_count'].mean():.0f}"
    )

    plt.axvline(
        corpus_df["paragraph_count"].median(),
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Median = {corpus_df['paragraph_count'].median():.0f}"
    )

    plt.title("Distribution of Paragraph Counts")
    plt.xlabel("Number of Paragraphs")
    plt.ylabel("Number of Judgments")

    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.show()
    return corpus_df


def compute_and_plot_average_words_per_paragraph(corpus_df):
    """
    7.6 Average Words per Paragraph

    The average number of words per paragraph provides an estimate of the semantic density of Supreme Court judgments. Since paragraphs generally represent coherent legal arguments, factual findings, or judicial reasoning, this measure is particularly useful for selecting an appropriate chunk size for retrieval-augmented generation.

    By examining the distribution of words across paragraphs, the analysis provides empirical evidence for determining the approximate number of paragraphs that can be accommodated within a single text chunk while preserving semantic coherence. Consequently, this metric serves as an important link between corpus analysis and the chunking strategy adopted in the NYAYaRAG framework.

    Observation:
    The average paragraph length across the cleaned corpus is 140.55 words, with a median of 141.90 words. Paragraph lengths range from approximately 42 to 203 words, indicating relatively limited variation compared with the substantial variability observed in document, sentence, and paragraph counts.

    The close agreement between the mean and median suggests that paragraph lengths are relatively stable across Supreme Court judgments despite considerable differences in overall document size. This consistency indicates that paragraphs represent well-defined semantic units within judicial decisions.

    From a retrieval perspective, these findings provide empirical support for the chunking strategy adopted in the NYAYaRAG framework. A text chunk of approximately 500–700 words would typically contain three to five complete paragraphs, thereby preserving semantic coherence while remaining within the input constraints of transformer-based embedding models. Consequently, the observed paragraph structure provides a practical basis for selecting an appropriate chunk size for embedding generation and retrieval.
    """
    corpus_df["avg_words_per_paragraph"] = (
        corpus_df["clean_words"] /
        corpus_df["paragraph_count"]
    )

    print("=" * 70)
    print("Average Words per Paragraph")
    print("=" * 70)

    print(f"Overall Average            : {corpus_df['avg_words_per_paragraph'].mean():.2f}")
    print(f"Median                     : {corpus_df['avg_words_per_paragraph'].median():.2f}")
    print(f"Minimum                    : {corpus_df['avg_words_per_paragraph'].min():.2f}")
    print(f"Maximum                    : {corpus_df['avg_words_per_paragraph'].max():.2f}")

    print("\nSummary Statistics\n")
    print(
        corpus_df["avg_words_per_paragraph"]
        .describe()
        .round(2)
        .to_frame(name="Value")
    )

    plt.figure(figsize=(10, 6), dpi=150)

    plt.hist(
        corpus_df["avg_words_per_paragraph"],
        bins=40,
        edgecolor="black",
        alpha=0.8
    )

    plt.axvline(
        corpus_df["avg_words_per_paragraph"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {corpus_df['avg_words_per_paragraph'].mean():.1f}"
    )

    plt.axvline(
        corpus_df["avg_words_per_paragraph"].median(),
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Median = {corpus_df['avg_words_per_paragraph'].median():.1f}"
    )

    plt.title("Distribution of Average Words per Paragraph")
    plt.xlabel("Average Words per Paragraph")
    plt.ylabel("Number of Judgments")

    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.show()
    return corpus_df


def compute_vocabulary_statistics(corpus_df):
    """
    7.7 Vocabulary Size

    Vocabulary size provides an indication of the lexical richness of the legal corpus by measuring the number of unique words appearing across all Supreme Court judgments. A larger vocabulary generally reflects greater linguistic diversity arising from multiple legal domains, statutory provisions, judicial terminology, and case-specific facts.

    Understanding the vocabulary of the corpus is important because retrieval models operate over textual representations of legal concepts. Examining the overall vocabulary therefore provides insight into the linguistic complexity of the corpus before embedding generation and retrieval.

    Observation:
    The cleaned corpus contains 6,492,583 word tokens and 47,961 unique words, resulting in a lexical diversity of 0.0074. The relatively low lexical diversity is expected in a specialised legal corpus, where judicial terminology, statutory references, procedural expressions, and legal phrases are repeatedly used across judgments.

    The most frequent tokens are common English function words such as the, of, to, and and. These high-frequency words primarily reflect the grammatical structure of judicial writing rather than domain-specific legal concepts. Consequently, stop words are excluded from subsequent frequency analyses to enable meaningful examination of the legal vocabulary represented in the corpus.
    """
    # Tokenize the entire cleaned corpus
    all_text = " ".join(corpus_df["clean_text"]).lower()

    tokens = re.findall(r"\b[a-zA-Z]+\b", all_text)

    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))

    vocabulary = Counter(tokens)

    print("=" * 70)
    print("Vocabulary Statistics")
    print("=" * 70)

    print(f"Total Tokens          : {total_tokens:,}")
    print(f"Unique Tokens         : {unique_tokens:,}")
    print(f"Lexical Diversity     : {unique_tokens / total_tokens:.4f}")

    print("\nTop 20 Most Frequent Words\n")
    print(
        pd.DataFrame(
            vocabulary.most_common(20),
            columns=["Word", "Frequency"]
        )
    )
    return vocabulary


def analyze_frequent_legal_terms(vocabulary):
    """
    7.8 Most Frequent Legal Terms

    After examining the overall vocabulary of the corpus, the analysis focuses on domain-specific legal terminology. Unlike common English function words, legal terms capture the substantive content of judicial decisions and provide insight into the legal concepts most frequently discussed within the corpus.

    Identifying frequently occurring legal terms helps characterise the thematic composition of the corpus and confirms that the dataset contains the specialised legal language required for developing and evaluating a legal retrieval-augmented generation (RAG) framework.

    Observation:
    The frequency analysis demonstrates that the corpus is dominated by domain-specific legal terminology rather than general language. The most frequently occurring legal terms include court (59,271), section (30,034), act (28,406), state (23,109), case (20,792), and order (19,350), reflecting the central role of statutory interpretation, judicial reasoning, and procedural adjudication within Supreme Court judgments.

    Frequently occurring terms such as appellant, respondent, appeal, evidence, accused, and conviction indicate substantial representation of appellate and criminal litigation, while terms including article, constitutional, rights, and government demonstrate the presence of constitutional and public law matters. The occurrence of property, authority, tribunal, and petition further illustrates the diversity of legal subject areas covered by the corpus.

    Overall, the distribution of legal terminology confirms that the dataset captures the specialised language of Indian judicial decisions, making it well suited for developing and evaluating a legal retrieval-augmented generation (RAG) framework.
    """
    legal_words = [
        "court",
        "section",
        "act",
        "state",
        "case",
        "order",
        "appellant",
        "respondent",
        "appeal",
        "article",
        "judge",
        "tribunal",
        "petition",
        "evidence",
        "constitutional",
        "accused",
        "government",
        "justice",
        "law",
        "authority",
        "police",
        "property",
        "rights",
        "offence",
        "conviction"
    ]

    legal_frequency = []

    for word in legal_words:
        legal_frequency.append(
            (
                word,
                vocabulary.get(word.lower(), 0)
            )
        )

    legal_df = (
        pd.DataFrame(
            legal_frequency,
            columns=["Legal Term", "Frequency"]
        )
        .sort_values("Frequency", ascending=False)
    )

    print("=" * 70)
    print("Most Frequent Legal Terms")
    print("=" * 70)
    print(legal_df)

    plt.figure(figsize=(10, 7), dpi=150)

    plt.barh(
        legal_df["Legal Term"],
        legal_df["Frequency"]
    )

    plt.xlabel("Frequency")
    plt.ylabel("Legal Term")
    plt.title("Most Frequent Legal Terms in the Corpus")

    plt.gca().invert_yaxis()

    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    plt.show()
    return legal_df


def analyze_frequently_mentioned_statutes(corpus_df):
    """
    7.10 Most Frequently Mentioned Statutes

    Beyond individual legal terms, the statutes cited within judicial decisions provide a direct indication of the legal domains represented in the corpus. Analysing statutory references enables identification of the areas of law most frequently considered by the Supreme Court and helps assess the thematic coverage of the dataset.

    Since the NYAYaRAG framework is intended for legal question answering, understanding the distribution of statutory references is important for evaluating whether the corpus encompasses diverse branches of Indian law, including constitutional, criminal, civil, commercial, regulatory, and tax-related legislation.
    """
    legal_terms = {

        # Constitutional Law
        "Constitution": r"\bConstitution\b|\bConstitution of India\b",

        # Criminal Law
        "Indian Penal Code (IPC)": r"\bIPC\b|\bIndian Penal Code\b",
        "Code of Criminal Procedure (CrPC)": r"\bCrPC\b|\bCode of Criminal Procedure\b",
        "Indian Evidence Act": r"\bEvidence Act\b|\bIndian Evidence Act\b",
        "NDPS Act": r"\bNDPS\b|\bNarcotic Drugs and Psychotropic Substances Act\b",
        "POCSO Act": r"\bPOCSO\b|\bProtection of Children from Sexual Offences Act\b",
        "SC/ST (Prevention of Atrocities) Act": r"\bSC/ST Act\b|\bScheduled Castes and Scheduled Tribes.*?Act\b",
        "Prevention of Corruption Act": r"\bPC Act\b|\bPrevention of Corruption Act\b",
        "National Investigation Agency Act": r"\bNIA Act\b|\bNational Investigation Agency Act\b",
        "Arms Act": r"\bArms Act\b",
        "Juvenile Justice Act": r"\bJuvenile Justice Act\b",
        "Drugs and Cosmetics Act": r"\bDrugs and Cosmetics Act\b",

        # Civil Law
        "Code of Civil Procedure (CPC)": r"\bCPC\b|\bCode of Civil Procedure\b",
        "Transfer of Property Act": r"\bTransfer of Property Act\b",
        "Specific Relief Act": r"\bSpecific Relief Act\b",
        "Contract Act": r"\bContract Act\b",
        "Limitation Act": r"\bLimitation Act\b",
        "Registration Act": r"\bRegistration Act\b",
        "Land Acquisition Act": r"\bLand Acquisition Act\b",

        # Family Law
        "Hindu Marriage Act": r"\bHindu Marriage Act\b",
        "Hindu Succession Act": r"\bHindu Succession Act\b",
        "Hindu Adoption and Maintenance Act": r"\bHindu Adoption and Maintenance Act\b",
        "Hindu Minority and Guardianship Act": r"\bHindu Minority and Guardianship Act\b",

        # Commercial Law
        "Companies Act": r"\bCompanies Act\b",
        "Arbitration and Conciliation Act": r"\bArbitration(?: and Conciliation)? Act\b",
        "Negotiable Instruments Act": r"\bNegotiable Instruments Act\b|\bNI Act\b",
        "SARFAESI Act": r"\bSARFAESI\b",
        "Insolvency and Bankruptcy Code": r"\bIBC\b|\bInsolvency and Bankruptcy Code\b",
        "Real Estate (Regulation and Development) Act": r"\bRERA\b|\bReal Estate \(Regulation and Development\) Act\b",

        # Tax
        "Income Tax Act": r"\bIncome Tax Act\b",
        "Customs Act": r"\bCustoms Act\b",
        "GST Act": r"\bGST Act\b|\bGoods and Services Tax Act\b",
        "Prevention of Money Laundering Act": r"\bPMLA\b|\bPrevention of Money Laundering Act\b",

        # Regulatory
        "Information Technology Act": r"\bIT Act\b|\bInformation Technology Act\b",
        "Electricity Act": r"\bElectricity Act\b",
        "Motor Vehicles Act": r"\bMotor Vehicles Act\b|\bMV Act\b",
        "Mines and Minerals (Development and Regulation) Act": r"\bMMDR Act\b|\bMines and Minerals \(Development and Regulation\) Act\b",
        "Consumer Protection Act": r"\bConsumer Protection Act\b",
        "Citizenship Act": r"\bCitizenship Act\b",

        # Labour
        "Industrial Disputes Act": r"\bIndustrial Disputes Act\b",
        "Employees' Compensation Act": r"\bEmployees'? Compensation Act\b",

        # Environment
        "Environment (Protection) Act": r"\bEnvironment \(Protection\) Act\b|\bEnvironment Protection Act\b",
        "Forest (Conservation) Act": r"\bForest Act\b|\bForest Conservation Act\b",

        # Election
        "Representation of the People Act": r"\bRepresentation of the People Act\b"
    }

    statute_counts = {}

    # Join the cleaned corpus
    full_text = " ".join(corpus_df["clean_text"])

    # Count occurrences of each statute
    for statute, pattern in legal_terms.items():
        matches = re.findall(pattern, full_text, flags=re.IGNORECASE)
        statute_counts[statute] = len(matches)

    # Create DataFrame
    statute_df = (
        pd.DataFrame(
            statute_counts.items(),
            columns=["Statute", "Frequency"]
        )
        .sort_values("Frequency", ascending=False)
        .reset_index(drop=True)
    )

    print("=" * 70)
    print("Most Frequently Mentioned Statutes")
    print("=" * 70)
    print(statute_df)

    top_statutes = statute_df.head(15)

    plt.figure(figsize=(12, 7), dpi=150)

    bars = plt.barh(
        top_statutes["Statute"],
        top_statutes["Frequency"]
    )

    # Add frequency labels
    for bar in bars:
        plt.text(
            bar.get_width() + 10,
            bar.get_y() + bar.get_height()/2,
            str(int(bar.get_width())),
            va="center",
            fontsize=9
        )

    plt.title("Top 15 Most Frequently Mentioned Statutes in Supreme Court Judgments (2024)")
    plt.xlabel("Frequency")
    plt.ylabel("Statute")

    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Alias Mapping
    alias_mapping = {

        # Criminal Law
        "NI Act": "Negotiable Instruments Act",
        "PC Act": "Prevention of Corruption Act",
        "PML Act": "Prevention of Money Laundering Act",
        "PMLA": "Prevention of Money Laundering Act",
        "IT Act": "Information Technology Act",
        "MV Act": "Motor Vehicles Act",
        "MMDR Act": "Mines and Minerals (Development and Regulation) Act",
        "NIA Act": "National Investigation Agency Act",
        "JJ Act": "Juvenile Justice Act",
        "UAPA": "Unlawful Activities (Prevention) Act",
        "SC/ST Act": "SC/ST (Prevention of Atrocities) Act",

        # Civil / Commercial
        "TP Act": "Transfer of Property Act",
        "Indian Contract Act": "Contract Act",
        "Indian Stamp Act": "Stamp Act",
        "Arbitration Act": "Arbitration and Conciliation Act",

        # Tax / Finance
        "CGST Act": "GST Act",

        # Public Law
        "RP Act": "Representation of the People Act",
        "RTI Act": "Right to Information Act",

        # Education
        "UGC Act": "University Grants Commission Act",
        "AMU Act": "Aligarh Muslim University Act",

        # Regulatory
        "PC&PNDT Act": "Pre-Conception and Pre-Natal Diagnostic Techniques Act",
        "DSPE Act": "Delhi Special Police Establishment Act",
        "CRPF Act": "Central Reserve Police Force Act",
        "MRTP Act": "Monopolies and Restrictive Trade Practices Act",

        # Environment
        "Forest Act": "Forest (Conservation) Act",

        # Duplicate forms
        "The Citizenship Act": "Citizenship Act",
        "Protection of Children from Sexual Offences Act": "POCSO Act",
        "Evidence Act": "Indian Evidence Act"
    }

    for alias, canonical in alias_mapping.items():

        count = len(
            re.findall(
                rf"\b{re.escape(alias)}\b",
                full_text,
                flags=re.IGNORECASE
            )
        )

        if canonical in statute_counts:
            statute_counts[canonical] += count

    # Create Final Canonical Statute Frequency Table
    statute_df = (
        pd.DataFrame(
            statute_counts.items(),
            columns=["Statute", "Frequency"]
        )
        .query("Frequency > 0")
        .sort_values(
            by="Frequency",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print("=" * 70)
    print("Final Canonical Statute Frequency Table")
    print("=" * 70)
    print(statute_df)

    top_statutes = statute_df.head(15).copy()

    plt.figure(figsize=(12, 7), dpi=150)

    bars = plt.barh(
        top_statutes["Statute"],
        top_statutes["Frequency"]
    )

    # Display frequency at the end of each bar
    for bar in bars:
        frequency = int(bar.get_width())

        plt.text(
            bar.get_width() + (top_statutes["Frequency"].max() * 0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{frequency:,}",
            va="center",
            fontsize=9
        )

    plt.title(
        "Top 15 Most Frequently Mentioned Statutes "
        "in the 2024 Supreme Court Judgment Corpus"
    )
    plt.xlabel("Number of Mentions")
    plt.ylabel("Statute")

    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)

    plt.xlim(
        0,
        top_statutes["Frequency"].max() * 1.10
    )

    plt.tight_layout()
    plt.show()
    return statute_df


def lexical_diversity(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    if len(words) == 0:
        return 0

    return len(set(words)) / len(words)


def compute_and_plot_lexical_diversity(corpus_df):
    """
    7.11 Lexical Diversity per Judgment

    While the overall corpus vocabulary provides a measure of lexical richness, lexical diversity at the individual judgment level indicates the extent to which each judgment employs a varied vocabulary. Lexical diversity is calculated as the ratio of unique words to the total number of words within a judgment.

    This analysis helps assess whether judicial decisions exhibit consistent linguistic complexity across the corpus or whether substantial variation exists between individual judgments.

    Observation:
    The distribution of lexical diversity is approximately unimodal with a slight positive skew, indicating that most judgments exhibit similar levels of vocabulary richness. The average lexical diversity is 0.180, while the median is 0.177, suggesting that vocabulary usage is relatively consistent across the corpus.

    Only a small number of judgments display substantially higher lexical diversity, likely reflecting lengthy Constitution Bench decisions or judgments addressing multiple legal issues and statutory frameworks. Overall, the relatively narrow distribution indicates that the corpus possesses a consistent linguistic style, which is desirable for developing and evaluating retrieval models and embedding-based legal question answering systems.
    """
    corpus_df["lexical_diversity"] = corpus_df["clean_text"].apply(
        lexical_diversity
    )

    print("=" * 70)
    print("Lexical Diversity per Judgment")
    print("=" * 70)

    print(f"Average Lexical Diversity : {corpus_df['lexical_diversity'].mean():.4f}")
    print(f"Median                    : {corpus_df['lexical_diversity'].median():.4f}")
    print(f"Minimum                   : {corpus_df['lexical_diversity'].min():.4f}")
    print(f"Maximum                   : {corpus_df['lexical_diversity'].max():.4f}")

    print("\nSummary Statistics\n")
    print(
        corpus_df["lexical_diversity"]
        .describe()
        .round(4)
        .to_frame(name="Value")
    )

    plt.figure(figsize=(10, 6), dpi=150)

    plt.hist(
        corpus_df["lexical_diversity"],
        bins=40,
        edgecolor="black"
    )

    plt.axvline(
        corpus_df["lexical_diversity"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {corpus_df['lexical_diversity'].mean():.3f}"
    )

    plt.axvline(
        corpus_df["lexical_diversity"].median(),
        color="green",
        linestyle=":",
        linewidth=2,
        label=f"Median = {corpus_df['lexical_diversity'].median():.3f}"
    )

    plt.title("Distribution of Lexical Diversity Across Judgments")

    plt.xlabel("Lexical Diversity")
    plt.ylabel("Number of Judgments")

    plt.grid(axis="y", alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.show()
    return corpus_df


def summarize_legal_corpus_characteristics(corpus_df, legal_df, statute_df):
    """
    7.12 Legal Corpus Characteristics

    The following table summarizes the key structural, linguistic, and legal characteristics of the Supreme Court judgment corpus after preprocessing. These statistics provide an overall profile of the dataset and establish the foundation for the subsequent chunking, embedding generation, and retrieval experiments in the NYAYaRAG framework.

    7.13 Summary

    The final cleaned corpus comprises 782 Supreme Court judgments, containing 6,820,528 words, 40,992,347 characters, 342,405 sentences, and 47,258 paragraphs. The median judgment contains 5,240.50 words, while judgment length ranges from 320 to 144,369 words, demonstrating substantial variation in document size.

    A total of 73 judgments (9.34%) were classified as length outliers using the Interquartile Range method. The corpus-wide average paragraph length is 144.33 words, providing an empirical basis for selecting a chunk size capable of preserving several semantically related paragraphs.

    The corpus contains 47,961 unique tokens. Its linguistic and legal characteristics are reflected in the frequent occurrence of the term court and the prominence of the Constitution of India among cited legal instruments. Overall, the corpus demonstrates sufficient scale, structural diversity, specialised legal vocabulary, and statutory coverage to support chunk generation, embedding construction, and comparative retrieval evaluation within the NYAYaRAG framework.
    """
    # Recalculate corpus-level vocabulary statistics
    all_tokens = re.findall(
        r"\b[a-zA-Z]+\b",
        " ".join(corpus_df["clean_text"]).lower()
    )

    vocabulary_size = len(set(all_tokens))

    corpus_lexical_diversity = (
        vocabulary_size / len(all_tokens)
        if len(all_tokens) > 0
        else 0
    )

    # Recalculate outlier values
    Q1 = corpus_df["clean_words"].quantile(0.25)
    Q3 = corpus_df["clean_words"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + (1.5 * IQR)

    outlier_count = (
        corpus_df["clean_words"] > upper_bound
    ).sum()

    outlier_percentage = (
        outlier_count / len(corpus_df) * 100
    )

    # Create summary table
    summary_table = pd.DataFrame({
        "Characteristic": [
            "Total Judgments",
            "Total Words",
            "Total Characters",
            "Average Words per Judgment",
            "Median Words per Judgment",
            "Minimum Words per Judgment",
            "Maximum Words per Judgment",
            "Average Characters per Judgment",
            "Total Sentences",
            "Average Sentences per Judgment",
            "Total Paragraphs",
            "Average Paragraphs per Judgment",
            "Average Words per Paragraph",
            "Vocabulary Size (Unique Tokens)",
            "Corpus Lexical Diversity",
            "Average Lexical Diversity per Judgment",
            "Judgment Length Outliers",
            "Most Frequent Legal Term",
            "Most Frequently Cited Statute"
        ],

        "Value": [
            f"{len(corpus_df):,}",
            f"{corpus_df['clean_words'].sum():,}",
            f"{corpus_df['clean_characters'].sum():,}",
            f"{corpus_df['clean_words'].mean():,.2f}",
            f"{corpus_df['clean_words'].median():,.2f}",
            f"{corpus_df['clean_words'].min():,}",
            f"{corpus_df['clean_words'].max():,}",
            f"{corpus_df['clean_characters'].mean():,.2f}",
            f"{corpus_df['sentence_count'].sum():,}",
            f"{corpus_df['sentence_count'].mean():,.2f}",
            f"{corpus_df['paragraph_count'].sum():,}",
            f"{corpus_df['paragraph_count'].mean():,.2f}",

            # Corpus-wide average words per paragraph
            f"{(
                corpus_df['clean_words'].sum()
                / corpus_df['paragraph_count'].sum()
            ):,.2f}",

            f"{vocabulary_size:,}",
            f"{corpus_lexical_diversity:.4f}",
            f"{corpus_df['lexical_diversity'].mean():.4f}",
            f"{outlier_count} ({outlier_percentage:.2f}%)",
            f"{legal_df.iloc[0]['Legal Term']} "
            f"({int(legal_df.iloc[0]['Frequency']):,})",
            f"{statute_df.iloc[0]['Statute']} "
            f"({int(statute_df.iloc[0]['Frequency']):,})"
        ]
    })

    print("=" * 70)
    print("Legal Corpus Characteristics")
    print("=" * 70)

    print(summary_table.to_string(index=False))
    return summary_table


def save_processed_metadata(corpus_df, output_path=r"D:\_ResearchNotebooks\NyayaRAG\data\processed\metadata_2024_processed.parquet"):
    """
    Save Processed Metadata (Parquet)
    """
    corpus_df.to_parquet(output_path, index=False)

    print("=" * 70)
    print("Processed metadata saved successfully.")
    print("=" * 70)
    print(output_path)


if __name__ == "__main__":
    PROJECT_DIR = Path(r"D:\_ResearchNotebooks\NyayaRAG")
    PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
    clean_corpus_file = PROCESSED_DIR / "clean_corpus.parquet"

    corpus_df = pd.read_parquet(clean_corpus_file)

    compute_corpus_statistics(corpus_df)
    plot_judgment_length_distribution(corpus_df)
    analyze_judgment_length_outliers(corpus_df)
    corpus_df = compute_and_plot_sentence_statistics(corpus_df)
    corpus_df = compute_and_plot_paragraph_statistics(corpus_df)
    corpus_df = compute_and_plot_average_words_per_paragraph(corpus_df)
    vocabulary = compute_vocabulary_statistics(corpus_df)
    legal_df = analyze_frequent_legal_terms(vocabulary)
    statute_df = analyze_frequently_mentioned_statutes(corpus_df)
    corpus_df = compute_and_plot_lexical_diversity(corpus_df)
    summarize_legal_corpus_characteristics(corpus_df, legal_df, statute_df)
    output_path = PROCESSED_DIR / "metadata_2024_processed.parquet"
    save_processed_metadata(corpus_df, str(output_path))
