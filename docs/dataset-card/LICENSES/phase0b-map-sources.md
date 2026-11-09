# Phase 0b proposed map layers: licenses and notices

The earlier Phase 0 approvals remain as amended. These are proposed new map
layers; the user still approves the five areas before production tracing.
Retain each source's own license and notices. Do not relabel the
third-party inputs as CC BY, ODbL or MIT merely because a benchmark
component uses that license.

| Layer | License and release treatment | Notice to carry |
| --- | --- | --- |
| Bavaria LoD2 and DGM1 | **CC BY 4.0**. [LoD2](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=lod2), [DGM1](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dgm1), [publisher attribution FAQ](https://www.geodaten.bayern.de/odd/m/3/html/faq.html), [license text](https://creativecommons.org/licenses/by/4.0/). Both clear the license screen. | Geobasisdaten: Bayerische Vermessungsverwaltung – [www.geodaten.bayern.de](https://www.geodaten.bayern.de/) (Daten verändert), Lizenz: CC BY 4.0. Use the unchanged-data variant for original copies. |
| 3DBAG, Dutch buildings | **CC BY 4.0**; [publisher download and credit](https://3dbag.bk.tudelft.nl/en/download), [API](https://api.3dbag.nl/api.html), [license text](https://creativecommons.org/licenses/by/4.0/). | 3DBAG by the 3D geoinformation research group and 3DGI, CC BY 4.0. Add the release, linked dataset, and changes (crop, triangulation, local coordinate transform and assigned materials). |
| AHN4 bare-earth DTM | **CC0 1.0** on the [distributed AHN4 dataset](https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.012026.28992.1). The [PDOK WCS](https://www.pdok.nl/ogc-webservices/-/article/actueel-hoogtebestand-nederland-ahn) links a [service record](https://www.nationaalgeoregister.nl/geonetwork/srv/api/records/bfcc588f-9393-4c70-b989-d9e92ac2f493) identifying AHN4 and marking unrestricted use under Public Domain Mark 1.0. This is the older 2020–2022 DTM, with the required 0.5 m native resolution. | No license attribution required for AHN4. For provenance use “Actueel Hoogtebestand Nederland (AHN4), DTM 0.5 m, via PDOK; cropped and regridded if applicable.” AHN5 is a separate asset with [a source-credit requirement](https://www.ahn.nl/tweede-deel-ahn5-beschikbaar); do not apply the AHN4 notice to it. |
| Berlin LoD2 and DGM1 | **Data licence Germany – Zero – Version 2.0**. [LoD2](https://daten.berlin.de/datensaetze/3d-gebaudemodelle-im-level-of-detail-2-lod-2-3c7c49af), [DGM](https://daten.berlin.de/datensaetze/atkis-dgm-digitales-gelandemodell-fa02f9e1), [license text](https://www.govdata.de/dl-de/zero-2-0). [GovData](https://www.govdata.de/informationen/lizenzen) describes Zero 2.0 as equivalent in effect to CC0. | No license attribution required. For provenance use “Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen Berlin; 3D-Gebäudemodelle LoD 2 / ATKIS DGM1; dl-de-zero-2.0; derived crop.” |
| Hamburg LoD2 and DGM1 (reserve) | **Data licence Germany – Attribution – Version 2.0**. [LoD2 catalog](https://suche.transparenz.hamburg.de/dataset/3d-stadtmodell-lod2-de-hamburg5), [DGM catalog](https://suche.transparenz.hamburg.de/dataset/digitales-hohenmodell-hamburg-dgm-16), [license](https://www.govdata.de/dl-de/by-2-0). [GovData](https://www.govdata.de/informationen/lizenzen) describes the attribution license as equivalent in effect to CC BY. | Freie und Hansestadt Hamburg, Landesbetrieb Geoinformation und Vermessung (LGV); Datenlizenz Deutschland – Namensnennung – Version 2.0; link the exact dataset and mark changes. Verify the notice of the actual selected vintage if this reserve is used. |
| OpenStreetMap fallback and LoD1 fidelity comparator | **ODbL 1.0**. [Copyright](https://www.openstreetmap.org/copyright), [ODbL](https://opendatacommons.org/licenses/odbl/1-0/), [attribution guidance](https://osmfoundation.org/wiki/Attribution_Guidelines). This is the separately approved OSM layer, not one of the permissively licensed LoD2 or DTM inputs. | © OpenStreetMap contributors, with the copyright link. Preserve extract date and geometry provenance, keep the OSM-derived database/geometry layer separable and meet applicable share-alike obligations. |

The recommended release arrangement is to retain provider credits and
terms on the individual source layers, publish the OSM-derived geometry
under ODbL, and decide a separate license for numeric NEC/channel outputs
after checking the contents for reconstructable geography. The current
audit does not choose the final benchmark license. DeepMIMO-hosted
scenarios remain on hold; the pinned DeepMIMO dependency is for pipeline
and format only.
