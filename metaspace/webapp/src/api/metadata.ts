import gql from 'graphql-tag';
import { datasetSubmitterFragment } from './user';

export const editDatasetFragment =
  gql`fragment EditDatasetFragment on Dataset {
    id
    metadataJson
    isPublic
    group {
      id
      name
    }
    projects {
      id
      name
    }
    principalInvestigator {
      name
      email
    }
    molDBs
    adducts
    name
  }`;

export const editDatasetQuery =
  gql`query fetchMetadataQuery($id: String!) {
    dataset(id: $id) {
      ...EditDatasetFragment
      submitter {
        ...DatasetSubmitterFragment
      }
    }
  }
  ${editDatasetFragment}
  ${datasetSubmitterFragment}
  `;

export const newDatasetQuery =
  gql`query {
    currentUserLastSubmittedDataset {
      ...EditDatasetFragment
    }
    currentUser {
      ...DatasetSubmitterFragment
    }
  }
  ${editDatasetFragment}
  ${datasetSubmitterFragment}
  `;

export const fetchAutocompleteSuggestionsQuery =
  gql`query suggestions($field: String!, $query: String!) {
    metadataSuggestions(field: $field, query: $query, limit: 5)
  }`;

export const updateDatasetQuery =
  gql`mutation ($id: String!, $reprocess: Boolean, $input: DatasetUpdateInput!) {
    updateDataset(id: $id, input: $input, reprocess: $reprocess priority: 1)
  }`;

// TODO: use autocompletion for filter values, same as on the upload page
export const fetchOptionListsQuery = gql`{
  institutionNames: metadataSuggestions(field: "Submitted_By.Institution", query: "", limit: 1000)
  organisms: metadataSuggestions(field: "Sample_Information.Organism", query: "", limit: 1000)
  organismParts: metadataSuggestions(field: "Sample_Information.Organism_Part", query: "", limit: 1000)
  conditions: metadataSuggestions(field: "Sample_Information.Condition", query: "", limit: 1000)
  growthConditions: metadataSuggestions(field: "Sample_Information.Sample_Growth_Conditions", query: "", limit: 1000)
  ionisationSources: metadataSuggestions(field: "MS_Analysis.Ionisation_Source", query: "", limit: 1000)
  maldiMatrices: metadataSuggestions(field: "Sample_Preparation.MALDI_Matrix", query: "", limit: 1000)
  analyzerTypes: metadataSuggestions(field: "MS_Analysis.Analyzer", query: "", limit: 1000)
  molecularDatabases: molecularDatabases(hideDeprecated: false){name, default}
  submitterNames: submitterSuggestions(query: "") {id, name}
  groups: allGroups {id, name}
  projects: allProjects {id, name}
  adducts: adductSuggestions{adduct, charge}
}`;

export const metadataOptionsQuery = gql`{
  molecularDatabases: molecularDatabases{name, default}
  adducts: adductSuggestions{adduct, charge}
}`;

export const metadataExportQuery = gql`
  query MetadataExport($dFilter: DatasetFilter, $offset: Int, $limit: Int,
                       $query: String, $inpFdrLvls: [Int!]!, $checkLvl: Int!) {
    datasets: allDatasets(filter: $dFilter, simpleQuery: $query,
                          offset: $offset, limit: $limit) {
      id
      name
      institution
      submitter {
        name
      }
      principalInvestigator {
        name
      }
      organism
      organismPart
      condition
      growthConditions
      ionisationSource
      maldiMatrix
      analyzer {
        type
        resolvingPower(mz: 400)
      }
      polarity
      uploadDateTime
      fdrCounts(inpFdrLvls: $inpFdrLvls, checkLvl: $checkLvl) {
        dbName
        levels
        counts
      }
      opticalImage
    }
  } `;
