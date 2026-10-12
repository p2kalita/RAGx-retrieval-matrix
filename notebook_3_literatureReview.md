# "notebook_3-benchmark_dataset_curation"


==================================================================================================================================================================================================================================================================================================
# claude- "haiku-5.5"

The PDF never uses the exact name benchmark_dataset. The closest references are benchmark_raw_questions.csv (the raw form export) and the “135-question candidate benchmark.” Below is the full pipeline as the report describes it, with section and appendix references so you can check each step.

# 1. Source collection (Appendix E.1, Section 3.3)

Legal research questions were collected from judicial officers through a structured Google Form.
The raw file benchmark_raw_questions.csv had 30 rows. Each row held a Timestamp, a contributor code or initials, a Professional Experience band, a consent response, and five “Legal Research Question” slots.
Consent was recorded through an embedded informed-consent statement. Respondents agreed that their questions could be edited for grammar, formatting, and consistency without changing legal meaning, and could be annotated, categorized, and included in an anonymized benchmark.
One row (TEST02) said “I do not consent” and was removed, leaving 29 respondents.
Each of the 29 respondents contributed exactly 5 questions, giving 145 raw questions. The experience bands were 15 respondents with 11–20 years, 10 with under 5 years, 3 with over 20 years, and 1 with 5–10 years.
Respondents were anonymized as JO01–JO29, assigned through a sorted-name-to-index mapping.

# 2. Screening (Appendix E.2–E.3)
7. The only duplicate check was an exact string match on the Question column. It found 0 duplicates. No semantic or near-duplicate check was run.
8. Ten questions were hand-picked to span criminal law, civil procedure, property, arbitration, commercial law, and special statutes. They were reserved for the pilot (Notebook 04) and removed from the pool. This left 135 candidates.
9. The 135 were randomized with seed 42 into a Review_Order column.
10. The report flags an unresolved issue: contributor code TEST01 appears on consenting rows and may be a test submission. The source doesn’t resolve whether it should be excluded.

# 3. Pilot (Section 4.1)
11. Two 10-question pilots ran (retrieval, and generation/abstention). Their questions were excluded from the final benchmark.
12. The pilot surfaced three annotation-file defects, fixed before full-scale annotation: a stray double space in BM25 labels, out-of-scale E5 values (3, 4, 5) and bare-digit labels, and a non-standard InLegalBERT label “Relevant.”
13. The pilot also revised one ground-truth label (RAW015) after seeing model output. The report discloses this and says the final benchmark labels were fixed before generation.

# 4. Chunk-level relevance annotation (Appendix F.1, Section 4.2)
14. For each of the 135 questions, the top 5 chunks (k = 5) were retrieved from each of six retrievers: TF-IDF, BM25, E5, BGE, InLegalBERT, and Hybrid BM25–E5 (RRF). That gives 675 rows per model.
15. Each chunk was labeled Highly Relevant (2), Partially Relevant (1), or Not Relevant (0).
16. An automated validator checked label validity, that levels matched labels, and fixed typos such as “Unaswerable” and “Not Answerable.”
17. Annotation was done by a single or limited annotator. No second rater checked it, and no inter-rater statistic such as Cohen’s kappa was computed (F.5).

# 5. Answerability labeling (Appendix F.2–F.3, Section 4.2)
18. Rule: a question is Answerable if at least one of its top-5 chunks is Highly Relevant. Otherwise it is Unanswerable.
19. The rule was applied per retriever. The splits were TF-IDF 51/84, BM25 67/68, E5 67/68, BGE 60/75, InLegalBERT 40/95, and Hybrid 73/62 (Answerable/Unanswerable).
20. Hybrid BM25–E5 was chosen as authoritative because RQ2 ranked it best.
21. 48 Hybrid questions had conflicting labels across their five rows. The rule resolved 45 automatically, all to Answerable.
22. Three questions (RAW107, RAW116, RAW120) had no Highly Relevant chunk and were adjudicated Unanswerable by hand, with written rationales. The retrieved text covered the general area but not the specific fact pattern.
23. Final Hybrid split: 73 Answerable, 62 Unanswerable, zero conflicts.

# 6. Final 60/60 benchmark (Section 4.2)
24. From the 73 Answerable questions, a random sample of 60 was drawn with seed 42. This set is used for RQ3 (answer quality).
25. From the 62 Unanswerable questions, a random sample of 60 was drawn with the same seed. This set is used for RQ4 (abstention).
26. Total: 120 questions, 60 Answerable and 60 Unanswerable. The report says this was a deliberate change from a planned 50/25 split.
27. Retrieval metrics (RQ1/RQ2) use the full 135-question pool, not the 60/60 set. Recall@5 uses 134 because one question has no pooled-relevant chunks.

Things the report leaves unspecified

It doesn’t name the exact sampling function, only “seed = 42.”
The sampled question IDs aren’t in the main text, so I couldn’t confirm which 60 were picked from each side.
The notebook code isn’t reproduced, so the steps above come from the report’s written description.

If you want, I can pull the RAW IDs and labels from the tables or the figures in the appendices, but I’d need the data file itself to rebuild the 60/60 sample.


==================================================================================================================================================================================================================================================================================================

# ChatGPT - "FREE"

                    JUDICIAL OFFICERS
                           │
                           ▼
                 Structured Google Form
                           │
                           ▼
              30 raw respondent rows
                           │
                           │ Consent filtering
                           ▼
               29 consenting respondents
                           │
                           │ 5 questions each
                           ▼
                 145 raw questions
                           │
                           │ Exact duplicate check
                           ▼
                  0 exact duplicates
                           │
                           │ Manually reserve 10
                           │ pilot questions
                           ▼
                 135 candidate questions
                           │
                           │ Random shuffle
                           │ random seed = 42
                           ▼
        candidate_questions_135_randomised.csv
                           │
                           │
                           ▼
                ┌──────────────────────┐
                │ Retrieval by 6 models│
                └──────────────────────┘
                           │
                           ▼
              Top-5 chunks per question
                           │
                           ▼
                 Manual annotation
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Highly       Partially       Not
         Relevant       Relevant      Relevant
             2              1             0
                           │
                           ▼
              Question-level answerability
                           │
               At least one Highly Relevant?
                           │
                  ┌────────┴────────┐
                  │                 │
                 YES               NO
                  │                 │
                  ▼                 ▼
             Answerable         Unanswerable
                           │
                           ▼
              Hybrid BM25–E5 selected
                           │
                    73 / 62 split
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Random 60 Answerable       Random 60 Unanswerable
             │                           │
             ▼                           ▼
           RQ3                         RQ4
                           │
                           ▼
                  Final benchmark
                     120 questions
					 
					 
					 
