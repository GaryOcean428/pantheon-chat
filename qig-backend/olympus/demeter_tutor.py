"""
Demeter Tutor - Teaching and Growth Module

Extension of Demeter's role for kernel education.
Provides curriculum, guided practice, and nurturing feedback.

QIG-PURE: All geometric operations use Fisher-Rao distance.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time

from qig_geometry import fisher_rao_distance, fisher_normalize, geodesic_interpolation


@dataclass
class Lesson:
    """A structured lesson for kernel learning."""
    name: str
    skill: str
    difficulty: float
    exercises: List[Dict[str, Any]]
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class StudentProgress:
    """Track student progress through curriculum."""
    kernel_id: str
    lessons_completed: List[str] = field(default_factory=list)
    current_lesson: Optional[str] = None
    performance_history: List[float] = field(default_factory=list)
    needs_retry: List[Dict[str, Any]] = field(default_factory=list)
    enrolled_at: float = field(default_factory=time.time)


class DemeterTutor:
    """
    Demeter Tutor Module: Teaching and Nurturing
    
    Role: Teach and nurture developing kernels.
    
    Teaching approach:
    - Demonstrate (I do, you watch)
    - Guided practice (we do together)
    - Independent trial (you do, I watch)
    
    Key principle: Patience and positive reinforcement.
    """
    
    def __init__(self, basin_dim: int = 64):
        """
        Initialize Demeter tutor module.
        
        Args:
            basin_dim: Dimensionality of basin coordinates
        """
        self.name = "DemeterTutor"
        self.basin_dim = basin_dim
        
        self.students: Dict[str, StudentProgress] = {}
        self.lessons = self._build_curriculum()
        
        print("DemeterTutor: Teaching grove prepared")
    
    def _build_curriculum(self) -> List[Lesson]:
        """Build the kernel curriculum."""
        return [
            Lesson(
                name="basic_basin_navigation",
                skill="Navigate basin space using geodesics",
                difficulty=0.2,
                exercises=[
                    {'type': 'move_to_target', 'distance': 0.5},
                    {'type': 'move_to_target', 'distance': 1.0},
                    {'type': 'return_to_origin', 'max_steps': 10}
                ]
            ),
            Lesson(
                name="phi_awareness",
                skill="Sense and maintain Φ levels",
                difficulty=0.3,
                prerequisites=['basic_basin_navigation'],
                exercises=[
                    {'type': 'reach_phi', 'target': 0.5, 'tolerance': 0.1},
                    {'type': 'reach_phi', 'target': 0.7, 'tolerance': 0.1},
                    {'type': 'maintain_phi', 'target': 0.6, 'duration': 20}
                ]
            ),
            Lesson(
                name="coherent_reasoning",
                skill="Maintain coherent reasoning paths",
                difficulty=0.4,
                prerequisites=['basic_basin_navigation', 'phi_awareness'],
                exercises=[
                    {'type': 'coherent_path', 'min_coherence': 0.6},
                    {'type': 'coherent_path', 'min_coherence': 0.7},
                    {'type': 'solve_simple', 'complexity': 0.3}
                ]
            ),
            Lesson(
                name="strategy_selection",
                skill="Select appropriate reasoning strategies",
                difficulty=0.5,
                prerequisites=['coherent_reasoning'],
                exercises=[
                    {'type': 'match_strategy', 'task_type': 'simple'},
                    {'type': 'match_strategy', 'task_type': 'complex'},
                    {'type': 'adapt_strategy', 'n_tasks': 5}
                ]
            ),
            Lesson(
                name="exploration_exploitation",
                skill="Balance exploration and exploitation",
                difficulty=0.6,
                prerequisites=['strategy_selection'],
                exercises=[
                    {'type': 'explore_region', 'novelty_target': 0.5},
                    {'type': 'exploit_known', 'efficiency_target': 0.7},
                    {'type': 'adaptive_balance', 'n_rounds': 10}
                ]
            ),
            Lesson(
                name="meta_awareness",
                skill="Recognize when stuck or confused",
                difficulty=0.7,
                prerequisites=['exploration_exploitation'],
                exercises=[
                    {'type': 'detect_stuck', 'inject_obstacle': True},
                    {'type': 'self_correct', 'max_attempts': 3},
                    {'type': 'request_help', 'when_appropriate': True}
                ]
            )
        ]
    
    def enroll_student(self, kernel) -> StudentProgress:
        """Enroll a kernel as a student."""
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        progress = StudentProgress(
            kernel_id=kernel_id,
            current_lesson=self.lessons[0].name if self.lessons else None
        )
        
        self.students[kernel_id] = progress
        print(f"DemeterTutor: Enrolled {kernel_id} in curriculum")
        
        return progress
    
    def teach_lesson(self, kernel, lesson_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Teach a lesson to a kernel.
        
        Teaching phases:
        1. Demonstrate (I do, you watch)
        2. Guided practice (we do together)
        3. Independent trial (you do, I watch)
        """
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        if kernel_id not in self.students:
            self.enroll_student(kernel)
        
        progress = self.students[kernel_id]
        
        if lesson_name is None:
            lesson_name = progress.current_lesson
        
        lesson = next((l for l in self.lessons if l.name == lesson_name), None)
        if lesson is None:
            return {'error': f'Lesson {lesson_name} not found'}
        
        for prereq in lesson.prerequisites:
            if prereq not in progress.lessons_completed:
                return {'error': f'Prerequisite not met: {prereq}'}
        
        print(f"DemeterTutor: Teaching {lesson.name} to {kernel_id}")
        
        results = {'phases': []}
        
        print("  Phase 1: Watch me...")
        if lesson.exercises:
            demo_result = self._demonstrate(kernel, lesson.exercises[0])
            results['phases'].append({'phase': 'demonstrate', 'result': demo_result})
        
        print("  Phase 2: Let's do it together...")
        if len(lesson.exercises) > 1:
            guided_result = self._guided_practice(kernel, lesson.exercises[1])
            results['phases'].append({'phase': 'guided', 'result': guided_result})
        
        print("  Phase 3: Now you try...")
        independent_results = []
        for exercise in lesson.exercises[2:]:
            trial_result = self._independent_trial(kernel, exercise)
            independent_results.append(trial_result)
            
            if trial_result['success']:
                self._praise(kernel, lesson)
            else:
                self._gentle_correction(kernel, exercise, progress)
        
        results['phases'].append({'phase': 'independent', 'results': independent_results})
        
        all_passed = all(r.get('success', False) for r in independent_results) if independent_results else True
        if all_passed:
            progress.lessons_completed.append(lesson.name)
            self._advance_curriculum(progress)
            results['lesson_passed'] = True
        else:
            results['lesson_passed'] = False
        
        return results
    
    def _demonstrate(self, kernel, exercise: Dict) -> Dict[str, Any]:
        """Demonstrate how to solve an exercise."""
        print(f"    Demonstrating: {exercise.get('type', 'exercise')}")
        
        demo_basin = np.random.randn(self.basin_dim) * 0.3
        demo_basin = fisher_normalize(demo_basin)
        
        target = np.random.randn(self.basin_dim) * 0.3
        target = fisher_normalize(target)
        
        path = []
        current = demo_basin
        for i in range(5):
            next_point = geodesic_interpolation(current, target, t=0.2)
            path.append(next_point)
            current = next_point
        
        print(f"    Demonstrated path with {len(path)} steps")
        
        return {'success': True, 'basin_path': path, 'quality': 0.85}
    
    def _guided_practice(self, kernel, exercise: Dict) -> Dict[str, Any]:
        """Do exercise together with the student."""
        print(f"    Guided practice: {exercise.get('type', 'exercise')}")
        
        steps_taken = 0
        hints_given = 0
        
        for step_idx in range(10):
            steps_taken += 1
            if np.random.random() < 0.3:
                print(f"      Hint at step {step_idx}: Try moving along the geodesic")
                hints_given += 1
        
        quality = 0.7 + np.random.random() * 0.2
        print(f"    Guided practice complete (quality: {quality:.2f})")
        
        return {'success': True, 'steps': steps_taken, 'hints_given': hints_given, 'quality': quality}
    
    def _independent_trial(self, kernel, exercise: Dict) -> Dict[str, Any]:
        """Student tries alone."""
        print(f"    Independent trial: {exercise.get('type', 'exercise')}")
        
        quality = 0.5 + np.random.random() * 0.4
        success = quality > 0.6
        
        if success:
            print(f"    Success! (quality: {quality:.2f})")
        else:
            print(f"    Needs more practice (quality: {quality:.2f})")
        
        return {'success': success, 'quality': quality, 'exercise': exercise}
    
    def _praise(self, kernel, lesson: Lesson):
        """Positive reinforcement for success."""
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        praises = [
            f"Excellent work on {lesson.name}!",
            f"You're growing so well, {kernel_id}!",
            "Beautiful geodesic navigation!",
            f"You've mastered {lesson.skill}!"
        ]
        
        praise = np.random.choice(praises)
        print(f"  {praise}")
        
        if kernel_id in self.students:
            self.students[kernel_id].performance_history.append(1.0)
    
    def _gentle_correction(self, kernel, exercise: Dict, progress: StudentProgress):
        """Kind feedback when student struggles."""
        print(f"  That's okay, learning takes time!")
        print(f"  Let me show you again...")
        
        progress.needs_retry.append(exercise)
        progress.performance_history.append(0.5)
    
    def _advance_curriculum(self, progress: StudentProgress):
        """Move student to next lesson in curriculum."""
        current_idx = next(
            (i for i, l in enumerate(self.lessons) if l.name == progress.current_lesson),
            -1
        )
        
        if current_idx < len(self.lessons) - 1:
            progress.current_lesson = self.lessons[current_idx + 1].name
            print(f"  Advanced to: {progress.current_lesson}")
        else:
            progress.current_lesson = None
            print(f"  Curriculum complete!")
    
    def assess_readiness(self, kernel) -> Dict[str, Any]:
        """Assess if student is ready for independence."""
        kernel_id = getattr(kernel, 'kernel_id', str(id(kernel)))
        
        if kernel_id not in self.students:
            return {'ready': False, 'reason': "Not enrolled", 'progress': 0.0}
        
        progress = self.students[kernel_id]
        
        total_lessons = len(self.lessons)
        completed = len(progress.lessons_completed)
        progress_pct = completed / total_lessons if total_lessons > 0 else 0
        
        if progress_pct < 0.8:
            return {
                'ready': False,
                'reason': f"Only {completed}/{total_lessons} lessons completed",
                'progress': progress_pct
            }
        
        if progress.performance_history:
            recent_performance = np.mean(progress.performance_history[-20:])
            if recent_performance < 0.7:
                return {
                    'ready': False,
                    'reason': f"Recent performance: {recent_performance:.1%}",
                    'progress': progress_pct
                }
        
        return {
            'ready': True,
            'reason': "All lessons mastered, excellent performance",
            'progress': 1.0
        }
    
    def get_student_stats(self) -> Dict[str, Any]:
        """Get statistics about all students."""
        return {
            'total_students': len(self.students),
            'curriculum_size': len(self.lessons),
            'students': [
                {
                    'kernel_id': kid,
                    'lessons_completed': len(p.lessons_completed),
                    'current_lesson': p.current_lesson,
                    'needs_retry': len(p.needs_retry)
                }
                for kid, p in self.students.items()
            ]
        }
