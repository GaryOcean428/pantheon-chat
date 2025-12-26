/**
 * Tool Factory Dashboard - main page component
 */

import React from 'react';
import { Wrench } from 'lucide-react';
import {
  ToolFilters,
  ToolList,
  ToolStats,
  ToolDetailPanel,
  ToolForm,
} from './components';
import { useToolFactory } from './hooks/useToolFactory';

export function ToolFactoryDashboard() {
  const {
    // Data
    tools,
    allTools,
    selectedTool,
    editingTool,
    isLoading,

    // UI State
    viewMode,
    searchQuery,
    selectedCategory,
    selectedStatus,
    sortField,
    sortDirection,
    isFormOpen,
    isSaving,
    isDeleting,

    // Actions
    setViewMode,
    setSearchQuery,
    setSelectedCategory,
    setSelectedStatus,
    setSelectedTool,
    toggleSort,
    openCreateForm,
    openEditForm,
    closeForm,
    handleSubmit,
    handleDelete,
  } = useToolFactory();

  return (
    <div className="container mx-auto py-6 space-y-6">
      <div className="flex items-center gap-3">
        <Wrench className="h-8 w-8 text-primary" />
        <div>
          <h1 className="text-3xl font-bold">Tool Factory</h1>
          <p className="text-muted-foreground">
            Create and manage QIG-powered tools
          </p>
        </div>
      </div>

      <ToolStats tools={allTools} />

      <ToolFilters
        searchQuery={searchQuery}
        selectedCategory={selectedCategory}
        selectedStatus={selectedStatus}
        viewMode={viewMode}
        onSearchChange={setSearchQuery}
        onCategoryChange={setSelectedCategory}
        onStatusChange={setSelectedStatus}
        onViewModeChange={setViewMode}
        onCreateClick={openCreateForm}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className={selectedTool ? 'lg:col-span-2' : 'lg:col-span-3'}>
          <ToolList
            tools={tools}
            viewMode={viewMode}
            isLoading={isLoading}
            sortField={sortField}
            sortDirection={sortDirection}
            onToolClick={setSelectedTool}
            onToolEdit={openEditForm}
            onSort={toggleSort}
          />
        </div>

        {selectedTool && (
          <div className="lg:col-span-1">
            <ToolDetailPanel
              tool={selectedTool}
              onClose={() => setSelectedTool(null)}
              onEdit={openEditForm}
              onDelete={handleDelete}
              isDeleting={isDeleting}
            />
          </div>
        )}
      </div>

      <ToolForm
        isOpen={isFormOpen}
        editingTool={editingTool}
        isSaving={isSaving}
        onClose={closeForm}
        onSubmit={handleSubmit}
      />
    </div>
  );
}
