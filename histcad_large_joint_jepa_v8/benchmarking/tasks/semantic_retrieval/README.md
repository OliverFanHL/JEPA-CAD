# Semantic retrieval

Reuses the saved frozen sentence-transformer targets and exact HistCAD split. Each representation gets
the same common-width semantic adapter and budget. Evaluation uses the same held-out query/gallery and
the exact existing study metrics: canonical and raw function matching, token Jaccard, dimensional
neighbor recall/error, structural matches, and the joint geometry-structure criterion.
