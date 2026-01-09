import React, { useEffect, useState } from 'react';

interface Sweep {
  id: string;
  status: string;
  metrics: any;
}

const Sweeps: React.FC = () => {
  const [sweeps, setSweeps] = useState<Sweep[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/sweeps')
      .then(res => res.json())
      .then(data => {
        setSweeps(data.sweeps || []);
        setLoading(false);
      });
  }, []);

  const handleAudit = (id: string) => {
    fetch(`/api/sweeps/${id}/audit`, { method: 'GET' });
  };

  const handleApprove = (id: string) => {
    fetch(`/api/sweeps/${id}/approve`, { method: 'POST' });
  };

  if (loading) return <div>Loading sweeps...</div>;

  return (
    <div>
      <h1>Sweeps Dashboard</h1>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {sweeps.map(sweep => (
            <tr key={sweep.id}>
              <td>{sweep.id}</td>
              <td>{sweep.status}</td>
              <td>
                <button onClick={() => handleAudit(sweep.id)}>Audit</button>
                <button onClick={() => handleApprove(sweep.id)}>Approve</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Sweeps;