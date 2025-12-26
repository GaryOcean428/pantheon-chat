/**
 * Tool Form - create/edit tool form
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  Input,
  Button,
  Label,
  Textarea,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  Badge,
} from '@/components/ui';
import { Plus, X, Trash2 } from 'lucide-react';
import { TOOL_CATEGORIES, PARAMETER_TYPES } from '../constants';
import type { Tool, ToolFormData, ToolParameter, ToolCategory } from '../types';

interface ToolFormProps {
  isOpen: boolean;
  editingTool: Tool | null;
  isSaving: boolean;
  onClose: () => void;
  onSubmit: (data: ToolFormData) => void;
}

const emptyParameter: ToolParameter = {
  name: '',
  type: 'string',
  required: false,
  description: '',
};

export function ToolForm({
  isOpen,
  editingTool,
  isSaving,
  onClose,
  onSubmit,
}: ToolFormProps) {
  const [name, setName] = useState(editingTool?.name ?? '');
  const [description, setDescription] = useState(editingTool?.description ?? '');
  const [category, setCategory] = useState<ToolCategory>(editingTool?.category ?? 'utility');
  const [parameters, setParameters] = useState<ToolParameter[]>(
    editingTool?.parameters ?? []
  );
  const [tags, setTags] = useState<string[]>(editingTool?.tags ?? []);
  const [tagInput, setTagInput] = useState('');

  React.useEffect(() => {
    if (isOpen) {
      setName(editingTool?.name ?? '');
      setDescription(editingTool?.description ?? '');
      setCategory(editingTool?.category ?? 'utility');
      setParameters(editingTool?.parameters ?? []);
      setTags(editingTool?.tags ?? []);
      setTagInput('');
    }
  }, [isOpen, editingTool]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      name,
      description,
      category,
      parameters: parameters.filter((p) => p.name.trim()),
      tags,
    });
  };

  const addParameter = () => {
    setParameters([...parameters, { ...emptyParameter }]);
  };

  const updateParameter = (index: number, updates: Partial<ToolParameter>) => {
    setParameters((prev) =>
      prev.map((p, i) => (i === index ? { ...p, ...updates } : p))
    );
  };

  const removeParameter = (index: number) => {
    setParameters((prev) => prev.filter((_, i) => i !== index));
  };

  const addTag = () => {
    const trimmed = tagInput.trim();
    if (trimmed && !tags.includes(trimmed)) {
      setTags([...tags, trimmed]);
      setTagInput('');
    }
  };

  const removeTag = (tag: string) => {
    setTags((prev) => prev.filter((t) => t !== tag));
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{editingTool ? 'Edit Tool' : 'Create New Tool'}</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Tool name"
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="category">Category</Label>
              <Select value={category} onValueChange={(v) => setCategory(v as ToolCategory)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TOOL_CATEGORIES.map((cat) => (
                    <SelectItem key={cat.value} value={cat.value}>
                      {cat.icon} {cat.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what this tool does"
              rows={3}
              required
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Parameters</Label>
              <Button type="button" variant="outline" size="sm" onClick={addParameter}>
                <Plus className="h-4 w-4 mr-1" />
                Add
              </Button>
            </div>
            {parameters.length > 0 && (
              <div className="space-y-3">
                {parameters.map((param, index) => (
                  <div key={index} className="flex gap-2 items-start p-3 rounded border">
                    <div className="flex-1 grid grid-cols-3 gap-2">
                      <Input
                        placeholder="Name"
                        value={param.name}
                        onChange={(e) => updateParameter(index, { name: e.target.value })}
                      />
                      <Select
                        value={param.type}
                        onValueChange={(v) =>
                          updateParameter(index, { type: v as ToolParameter['type'] })
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {PARAMETER_TYPES.map((type) => (
                            <SelectItem key={type.value} value={type.value}>
                              {type.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Input
                        placeholder="Description"
                        value={param.description}
                        onChange={(e) =>
                          updateParameter(index, { description: e.target.value })
                        }
                      />
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => removeParameter(index)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="space-y-2">
            <Label>Tags</Label>
            <div className="flex gap-2">
              <Input
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                placeholder="Add tag"
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addTag();
                  }
                }}
              />
              <Button type="button" variant="outline" onClick={addTag}>
                Add
              </Button>
            </div>
            {tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="gap-1">
                    {tag}
                    <button
                      type="button"
                      onClick={() => removeTag(tag)}
                      className="hover:text-destructive"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={isSaving || !name.trim() || !description.trim()}>
              {isSaving ? 'Saving...' : editingTool ? 'Update' : 'Create'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
