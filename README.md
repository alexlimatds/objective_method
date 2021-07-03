# Objective method

This project aims to develop an automated implementation of the objective method, which aims to aid researchers in setting a query string for a Systematic Literature Review (SLR). While the original method is designed to the biomedical domain, this projetc targets the computer science domain and, therefore, some changes were required.

The current implementation is funcional but few evaluations were peformed, which showed poor results. A cleary improvent comes from the term categorization mechanism, which does not discard terms. In doing that, trash terms are being included in the final query.

The objective method is decribed in the following papers:

 - Hausner, E., Waffenschmidt, S., Kaiser, T. et al. Routine development of objectively derived search strategies. Syst Rev 1, 19 (2012). https://doi.org/10.1186/2046-4053-1-19
 - Simon M, Hausner E, Klaus SF, Dunton NE. Identifying nurse staffing research in Medline: development and testing of empirically derived search strategies with the PubMed interface. BMC Med Res Methodol. 2010 Aug 23;10:76. doi: 10.1186/1471-2288-10-76. PMID: 20731858; PMCID: PMC2936389.

An implementation of the objective method is presented in:

- Scells H., Zuccon G., Koopman B., Clark J. (2020) A Computational Approach for Objectively Derived Systematic Review Search Strategies. In: Jose J. et al. (eds) Advances in Information Retrieval. ECIR 2020. Lecture Notes in Computer Science, vol 12035. Springer, Cham. https://doi.org/10.1007/978-3-030-45439-5_26 ([preprint](https://scells.me/publication/ecir2020_objective/))
