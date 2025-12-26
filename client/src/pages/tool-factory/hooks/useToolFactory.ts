/**
 * Hook for Tool Factory state management
 */

import { useState, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import type { Tool, ToolFormData, ViewMode, SortField, SortDirection } from '../types';

interface UseToolFactoryOptions {
  initialViewMode?: ViewMode;
}

const fetchTools = async (): Promise<Tool[]> => {
  const response = await fetch('/api/tools');
  if (!response.ok) throw new Error('Failed to fetch tools');
  return response.json();
};

const createTool = async (data: ToolFormData): Promise<Tool> => {
  const response = await fetch('/api/tools', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error('Failed to create tool');
  return response.json();
};

const updateTool = async ({ id, data }: { id: string; data: Partial<ToolFormData> }): Promise<Tool> => {
  const response = await fetch(`/api/tools/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error('Failed to update tool');
  return response.json();
};

const deleteTool = async (id: string): Promise<void> => {
  const response = await fetch(`/api/tools/${id}`, { method: 'DELETE' });
  if (!response.ok) throw new Error('Failed to delete tool');
};

export function useToolFactory(options: UseToolFactoryOptions = {}) {
  const { initialViewMode = 'grid' } = options;
  const queryClient = useQueryClient();

  // UI State
  const [viewMode, setViewMode] = useState<ViewMode>(initialViewMode);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedStatus, setSelectedStatus] = useState<string | null>(null);
  const [sortField, setSortField] = useState<SortField>('updatedAt');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingTool, setEditingTool] = useState<Tool | null>(null);

  // Queries
  const toolsQuery = useQuery({
    queryKey: ['tools'],
    queryFn: fetchTools,
  });

  // Mutations
  const createMutation = useMutation({
    mutationFn: createTool,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tools'] });
      setIsFormOpen(false);
    },
  });

  const updateMutation = useMutation({
    mutationFn: updateTool,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tools'] });
      setEditingTool(null);
      setIsFormOpen(false);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteTool,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tools'] });
      setSelectedTool(null);
    },
  });

  // Filtered and sorted tools
  const filteredTools = useMemo(() => {
    let tools = toolsQuery.data ?? [];

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      tools = tools.filter(
        (tool) =>
          tool.name.toLowerCase().includes(query) ||
          tool.description.toLowerCase().includes(query) ||
          tool.tags.some((tag) => tag.toLowerCase().includes(query))
      );
    }

    if (selectedCategory) {
      tools = tools.filter((tool) => tool.category === selectedCategory);
    }

    if (selectedStatus) {
      tools = tools.filter((tool) => tool.status === selectedStatus);
    }

    tools = [...tools].sort((a, b) => {
      let comparison = 0;
      switch (sortField) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'usageCount':
          comparison = a.usageCount - b.usageCount;
          break;
        case 'successRate':
          comparison = a.successRate - b.successRate;
          break;
        case 'updatedAt':
          comparison = new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime();
          break;
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });

    return tools;
  }, [toolsQuery.data, searchQuery, selectedCategory, selectedStatus, sortField, sortDirection]);

  // Actions
  const openCreateForm = useCallback(() => {
    setEditingTool(null);
    setIsFormOpen(true);
  }, []);

  const openEditForm = useCallback((tool: Tool) => {
    setEditingTool(tool);
    setIsFormOpen(true);
  }, []);

  const closeForm = useCallback(() => {
    setIsFormOpen(false);
    setEditingTool(null);
  }, []);

  const handleSubmit = useCallback(
    (data: ToolFormData) => {
      if (editingTool) {
        updateMutation.mutate({ id: editingTool.id, data });
      } else {
        createMutation.mutate(data);
      }
    },
    [editingTool, createMutation, updateMutation]
  );

  const handleDelete = useCallback(
    (tool: Tool) => {
      if (confirm(`Are you sure you want to delete "${tool.name}"?`)) {
        deleteMutation.mutate(tool.id);
      }
    },
    [deleteMutation]
  );

  const toggleSort = useCallback((field: SortField) => {
    setSortField((prev) => {
      if (prev === field) {
        setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'));
        return prev;
      }
      setSortDirection('desc');
      return field;
    });
  }, []);

  return {
    // Data
    tools: filteredTools,
    allTools: toolsQuery.data ?? [],
    selectedTool,
    editingTool,
    isLoading: toolsQuery.isLoading,
    error: toolsQuery.error,

    // UI State
    viewMode,
    searchQuery,
    selectedCategory,
    selectedStatus,
    sortField,
    sortDirection,
    isFormOpen,
    isSaving: createMutation.isPending || updateMutation.isPending,
    isDeleting: deleteMutation.isPending,

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
  };
}
