# Security policy

Report suspected vulnerabilities privately by emailing
<hernan.grecco@gmail.com> with `fpm-rs security` in the subject. Do not include
an unpatched vulnerability in a public issue. Include the affected version,
impact, reproduction steps, and any proposed mitigation; receipt will be
acknowledged before public disclosure is coordinated.

Reports involving dataset registry discovery, checksummed network downloads,
archive validation or extraction, managed caching, parsers, and generated
artifacts are in scope. A checksum establishes content identity but does not by
itself make untrusted archive contents safe.

Security fixes target the current release line and `main`; older releases may
require upgrading.
