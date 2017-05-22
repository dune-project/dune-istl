/* begin dune-istl
   put the definitions for config.h specific to
   your project here. Everything above will be
   overwritten
*/

/* begin private */
/* Name of package */
#define PACKAGE "@DUNE_MOD_NAME@"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "@DUNE_MAINTAINER@"

/* Define to the full name of this package. */
#define PACKAGE_NAME "@DUNE_MOD_NAME@"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "@DUNE_MOD_NAME@ @DUNE_MOD_VERSION@"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "@DUNE_MOD_NAME@"

/* Define to the home page for this package. */
#define PACKAGE_URL "@DUNE_MOD_URL@"

/* Define to the version of this package. */
#define PACKAGE_VERSION "@DUNE_MOD_VERSION@"

/* end private */

/* Define to ENABLE_SUPERLU if the SuperLU library is available */
#cmakedefine HAVE_SUPERLU ENABLE_SUPERLU

/* Define to the integer type that SuperLU was compiled for
   See e.g. what int_t is defined to in slu_sdefs.h */
#cmakedefine SUPERLU_INT_TYPE @SUPERLU_INT_TYPE@

/* Define to 1 if header slu_sdefs.h is there. */
#cmakedefine01 HAVE_SLU_SDEFS_H

/* Define to 1 if header slu_ddefs.h is there. */
#cmakedefine01 HAVE_SLU_DDEFS_H

/* Define to 1 if header slu_cdefs.h is there. */
#cmakedefine01 HAVE_SLU_CDEFS_H

/* Define to 1 if header slu_zdefs.h is there. */
#cmakedefine01 HAVE_SLU_ZDEFS_H

/* Define to ENABLE_ARPACKPP if the ARPACK++ library is available */
#cmakedefine HAVE_ARPACKPP ENABLE_ARPACKPP

/* Define to 0 as all versions since SuperLu 4.0 do no longer provide it that way. */
#define HAVE_MEM_USAGE_T_EXPANSIONS 1

/* define to 1 if SuperLU header slu_ddefs.h contains SLU_DOUBLE */
#cmakedefine SUPERLU_MIN_VERSION_4_3 @SUPERLU_MIN_VERSION_4_3@

/* define to 1 if SuperLU dgssvx takes a GlobalLU_t parameter */
#cmakedefine SUPERLU_MIN_VERSION_5 @SUPERLU_MIN_VERSION_5@

/* Define to the version of dune-istl */
#define DUNE_ISTL_VERSION "${DUNE_ISTL_VERSION}"

/* Define to the major version of dune-istl */
#define DUNE_ISTL_VERSION_MAJOR ${DUNE_ISTL_VERSION_MAJOR}

/* Define to the minor version of dune-istl */
#define DUNE_ISTL_VERSION_MINOR ${DUNE_ISTL_VERSION_MINOR}

/* Define to the revision of dune-istl */
#define DUNE_ISTL_VERSION_REVISION ${DUNE_ISTL_VERSION_REVISION}

/* Enable/Disable the backwards compatibility of the category enum/method in dune-istl solvers, preconditioner, etc. */
#cmakedefine DUNE_ISTL_SUPPORT_OLD_CATEGORY_INTERFACE @DUNE_ISTL_SUPPORT_OLD_CATEGORY_INTERFACE@

/* end dune-istl
   Everything below here will be overwritten
*/
