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

/* define if the Boost::Fusion headers are available */
#cmakedefine HAVE_BOOST_FUSION

/* Define to ENABLE_BOOST if the Boost library is available */
#cmakedefine HAVE_BOOST ENABLE_BOOST

/* Define to ENABLE_SUPERLU if the SuperLU library is available */
#cmakedefine HAVE_SUPERLU ENABLE_SUPERLU

/* define to 1 because older versions of SuperLU are no longer supported*/
#define SUPERLU_POST_2005_VERSION 1

/* Define to 1 if 'expansions' is a member of 'mem_usage_t'. */
#cmakedefine HAVE_MEM_USAGE_T_EXPANSIONS @HAVE_MEM_USAGE_T_EXPANSIONS@

/* define to 1 if SuperLU header slu_ddefs.h contains SLU_DOUBLE */
#cmakedefine SUPERLU_MIN_VERSION_4_3 @SUPERLU_MIN_VERSION_4_3@

/* Define to the version of dune-istl */
#define DUNE_ISTL_VERSION "${DUNE_ISTL_VERSION}"

/* Define to the major version of dune-istl */
#define DUNE_ISTL_VERSION_MAJOR ${DUNE_ISTL_VERSION_MAJOR}

/* Define to the minor version of dune-istl */
#define DUNE_ISTL_VERSION_MINOR ${DUNE_ISTL_VERSION_MINOR}

/* Define to the revision of dune-istl */
#define DUNE_ISTL_VERSION_REVISION ${DUNE_ISTL_VERSION_REVISION}

/* end dune-istl
   Everything below here will be overwritten
*/
