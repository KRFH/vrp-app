# 人材制約の実装

## 概要

OR-Toolsを使用したCVRPTWソルバーに人材制約（ワーカーのスキル制約）を組み込みました。これにより、各ノード（顧客）に必要なスキルを持つワーカーのみが訪問できるようになります。

## 実装のポイント

### 1. なぜ複雑なのか

OR-ToolsのRoutingModelは「車両（vehicle）」単位でルートを決めますが、人材制約を実装するには：

- 車両 = ワーカーとみなす必要がある
- 各車両（ワーカー）に「できる仕事（スキル）」を持たせる必要がある
- 各ノード（顧客）に「必要スキル」を持たせる必要がある
- RoutingModelの標準APIには「この車両はこのノードに行ける／行けない」という制約を直接書く方法が少ない

### 2. 解決策

`SetAllowedVehiclesForIndex`を使用して、各ノードに対して訪問可能な車両（ワーカー）を制限する方法を採用しました。

## 実装詳細

### モデルの拡張

#### `models.py`の変更

```python
@dataclass
class Worker:
    """Worker with skills information."""
    id: int
    name: str
    skills: Set[str]

@dataclass
class Node:
    # ... 既存のフィールド ...
    required_skills: Set[str] = None  # required skills for this node

    def __post_init__(self):
        if self.required_skills is None:
            self.required_skills = {self.task} if self.task else set()
```

#### `create_data_model`関数の変更

```python
def create_data_model(nodes: List[Node], workers: List[Worker], cap: int) -> Dict:
    """Create data model for CVRPTW solver with worker constraints."""
    return {
        # ... 既存のフィールド ...
        "required_skills": [n.required_skills for n in nodes],
        "vehicle_capacities": [cap] * len(workers),
        "num_vehicles": len(workers),
        "workers": workers,
        "depot": 0,
    }
```

### OR-Toolsソルバーの拡張

#### `solvers/ortools_solver.py`の変更

```python
def _define_constraints(self) -> None:
    """Define all problem constraints."""
    self._define_time_constraints()
    self._define_capacity_constraints()
    self._define_time_window_constraints()
    self._define_worker_constraints()  # 新規追加

def _define_worker_constraints(self) -> None:
    """Define worker skill constraints using SetAllowedVehiclesForIndex."""
    # Get worker skills and node requirements
    workers = self.data.get("workers", [])
    required_skills = self.data.get("required_skills", [])
    
    # For each node, determine which vehicles (workers) can visit it
    for node_idx, node_skills in enumerate(required_skills):
        if not node_skills:  # No skills required, all vehicles can visit
            continue
            
        # Find vehicles (workers) that have the required skills
        allowed_vehicles = []
        for vehicle_id, worker in enumerate(workers):
            if node_skills.issubset(worker.skills):
                allowed_vehicles.append(vehicle_id)
        
        # Set allowed vehicles for this node
        if allowed_vehicles:
            routing_idx = self.mgr.NodeToIndex(node_idx)
            self.routing.SetAllowedVehiclesForIndex(allowed_vehicles, routing_idx)
```

### ワーカー情報の解析

#### `worker_assignment.py`の変更

```python
def parse_workers(text: str, veh_count: int) -> List[Worker]:
    """Parse worker information from text input."""
    workers = []
    if text:
        for part in text.split(','):
            part = part.strip()
            if not part:
                continue
            if ':' in part:
                name, skill_str = part.split(':', 1)
                skills = {s.strip() for s in skill_str.split('|') if s.strip()}
            else:
                name = part
                skills = set()
            workers.append(Worker(len(workers), name.strip(), skills))
    
    # Fill remaining slots with default workers
    while len(workers) < veh_count:
        workers.append(Worker(len(workers), f"Worker {len(workers)}", set()))
    
    return workers[:veh_count]
```

## 使用方法

### 1. ワーカー情報の入力

Webアプリケーションでは、以下の形式でワーカー情報を入力します：

```
A:delivery, B:repair, C:maintenance
```

または、複数のスキルを持つワーカー：

```
A:delivery|repair, B:maintenance, C:delivery
```

### 2. ノードのスキル要件

各ノード（顧客）には、以下のいずれかの方法でスキル要件を設定できます：

- `task`フィールドに単一のスキルを設定（例：`"delivery"`）
- `required_skills`フィールドに複数のスキルを設定（例：`{"delivery", "repair"}`）

### 3. 制約の動作

- 各ノードは、そのノードの`required_skills`をすべて持つワーカーのみが訪問可能
- スキル要件がないノード（空集合）は、すべてのワーカーが訪問可能
- デポ（ノード0）は制約なし

## テスト

`test_worker_constraints.py`を実行して、実装が正しく動作することを確認できます：

```bash
python test_worker_constraints.py
```

## 制約の効果

この実装により：

1. **スキルマッチング**: 各ノードは適切なスキルを持つワーカーのみが訪問
2. **最適化**: OR-Toolsが制約を考慮して最適なルートを計算
3. **柔軟性**: 複数のスキル要件や複合スキルにも対応
4. **効率性**: 制約がソルバー内で直接処理されるため、後処理が不要

## 今後の拡張可能性

- スキルレベルの追加（初級、中級、上級など）
- 時間帯によるスキル制約
- チーム作業の制約
- スキル習得コストの考慮 