<template>
  <tag-filter name="Class" removable @destroy="destroy">
    <template v-slot:edit>
      <el-select
        :model-value="filterValues.molClass ? parseInt(filterValues.molClass, 10) : undefined"
        placeholder="Select molecular ontology"
        filterable
        clearable
        :teleported="false"
        remote
        @change="(val) => onChange('molClass', val)"
      >
        <el-option
          v-for="item in molClasses"
          :key="item.id"
          :label="`${item.name} - ${item.molType} - ${item.category}`"
          :value="item.id"
        />
      </el-select>
      <el-select
        :model-value="filterValues.term ? parseInt(filterValues.term, 10) : undefined"
        :remote-method="updateTermQuery"
        :loading="termOptionsLoading"
        placeholder="Select term"
        filterable
        clearable
        remote
        @focus="
          () => {
            updateTermQuery('')
          }
        "
        @change="(val) => onChange('term', val)"
      >
        <el-option v-for="item in termOptions" :key="item.id" :label="item.enrichmentName" :value="item.id" />
      </el-select>
    </template>
    <template v-slot:show>
      <span class="tf-value-span">
        <span>{{ formatValue() }}</span>
      </span>
    </template>
  </tag-filter>
</template>

<script lang="ts">
import { defineComponent, computed, reactive } from 'vue'
import { useQuery } from '@vue/apollo-composable'
import gql from 'graphql-tag'
import TagFilter from './TagFilter.vue'

export default defineComponent({
  name: 'ClassFilter',
  components: {
    TagFilter,
  },
  props: {
    filterValues: Object as any,
  },
  setup(props, { emit }) {
    const state = reactive({
      termNameQuery: '',
      ontoId: props.filterValues.molClass,
      termId: props.filterValues.term,
    })

    const ENRICHMENT_DATABASES_QUERY = gql`
      query EnrichmentDatabases {
        allEnrichmentDatabases {
          id
          name
          molType
          category
        }
      }
    `
    const ENRICHMENT_TERMS_QUERY = gql`
      query EnrichmentTerms($databaseId: Int, $id: Int, $enrichmentName: String) {
        allEnrichmentTerms(databaseId: $databaseId, id: $id, enrichmentName: $enrichmentName) {
          id
          enrichmentName
        }
      }
    `

    const { result: molClassesResult } = useQuery(ENRICHMENT_DATABASES_QUERY)
    const { result: termOptionsResult, loading: termOptionsLoading } = useQuery(
      ENRICHMENT_TERMS_QUERY,
      computed(() => ({
        databaseId: parseInt(state.ontoId, 10),
        id: state.termId ? parseInt(state.termId, 10) : undefined,
        enrichmentName: state.termNameQuery,
      }))
    )
    const molClasses: any = computed(() => molClassesResult.value?.allEnrichmentDatabases)
    const termOptions: any = computed(
      () => termOptionsResult.value?.allEnrichmentTerms || [{ id: -1, enrichmentName: 'No terms' }]
    )

    const formatValue = () => {
      const { molClass, term } = props.filterValues
      const classItem = (molClasses.value || []).find((item: any) => item.id === parseInt(molClass, 10))
      const termItem = (termOptions.value || []).find((item: any) => item.id === parseInt(term, 10))

      if (classItem && termItem) {
        return `${classItem.name} - ${termItem.enrichmentName}`
      } else if (classItem) {
        return classItem.name
      } else {
        return '(Any)'
      }
    }

    const updateTermQuery = (query: string) => {
      state.termNameQuery = query
      state.termId = undefined
    }

    const onChange = (filterKey: 'molClass' | 'term', val: any) => {
      if (val) {
        if (filterKey === 'molClass') {
          updateTermQuery('')
          emit('change', null, 'term')
        }
        emit('change', val, filterKey)
      } else {
        emit('destroy', filterKey)
      }
    }

    const destroy = () => {
      emit('destroy', 'molClass')
      emit('destroy', 'term')
    }

    return {
      molClasses,
      termOptions,
      termOptionsLoading,
      updateTermQuery,
      formatValue,
      onChange,
      destroy,
    }
  },
})
</script>

<style scoped>
.el-select {
  width: 100%;
}
</style>
