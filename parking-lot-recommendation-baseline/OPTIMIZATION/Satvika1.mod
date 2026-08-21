/*********************************************
 * OPL 12.8.0.0 Model
 * Dynamic Parking Optimization Framework
 * Strengthened version without extra data fields
 *********************************************/

// Parameters
int Nodes = ...;     
range N = 1..Nodes;

tuple edge {
  int Source;
  int Destination;
}
{edge} E = ...;

int S = ...;
int D = ...;

int Q = ...;
range P = 1..Q;
int L[P] = ...;
float C[P] = ...;

int B = ...;
float Alpha = ...;
float W_km = ...;
float Cmax = ...;

float Distance[E] = ...;
float EdgeTime[E] = ...;

// Decision Variables
dvar boolean X[E];
dvar boolean XP[P];
dvar float+ T[N];
dvar float+ Cost;

// Objective: weighted non-normalized travel time in minutes plus parking cost
minimize Alpha * (60 * T[D]) + (1 - Alpha) * Cost;

subject to {

  /***********************
   * Parking selection
   ***********************/
  sum(p in P) XP[p] == 1;

  forall(p in P)
    X[<L[p], D>] == XP[p];

  Cost == sum(p in P) C[p] * XP[p];

  Cost <= Cmax;

  forall(p in P)
    Distance[<L[p], D>] * XP[p] <= W_km;

  /***********************
   * Route structure
   ***********************/

  // Source: exactly one outgoing, no incoming
  sum(<i,j> in E : i == S) X[<i,j>] == 1;
  sum(<j,i> in E : i == S) X[<j,i>] == 0;

  // Destination: exactly one incoming, no outgoing
  sum(<j,i> in E : i == D) X[<j,i>] == 1;
  sum(<i,j> in E : i == D) X[<i,j>] == 0;

  // Intermediate-node flow conservation
  forall(j in N : j != S && j != D)
    sum(<i,j> in E) X[<i,j>] - sum(<j,i> in E) X[<j,i>] == 0;

  // Degree caps
  forall(i in N : i != D)
    sum(<i,j> in E) X[<i,j>] <= 1;

  forall(j in N : j != S)
    sum(<i,j> in E) X[<i,j>] <= 1;

  // Reverse-edge blocking
  forall(<i,j> in E : <j,i> in E)
    X[<i,j>] + X[<j,i>] <= 1;

  /***********************
   * Time propagation
   ***********************/
  T[S] == 0;

  forall(<h,i> in E : i != S)
    (X[<h,i>] == 1) => (T[i] == T[h] + EdgeTime[<h,i>]);
}

execute
{
    var DistanceTaken = 0;

    writeln(">>> EV Routing >>>");
    for (var e in E)
    {
        if (X[e] > 0)
        {
            writeln(">>> X[", e.Source, "->", e.Destination, "] ");
            DistanceTaken = DistanceTaken + Distance[e];
        }
    }
    writeln();

    writeln("#### EV Arrival Time ####");
    writeln("#### T[", S, "] =  ", T[S]);
    for (var n in N)
    {
        for (var e in E)
        {
            if (e.Destination == n && X[e] == 1)
                writeln("#### T[", n, "] =  ", T[n]);
        }
    }
    writeln();

    writeln("#### Selected Parking Lot ####");
    for (var p in P)
    {
        if (XP[p] == 1)
        {
            writeln("Parking lot index: ", p);
            writeln("Parking node: ", L[p]);
            writeln("Parking cost: ", C[p]);
            writeln("Driving time to parking lot: ", T[L[p]]);

            for (var e in E)
            {
                if (e.Source == L[p] && e.Destination == D)
                {
                    writeln("Walking distance to destination: ", Distance[e]);
                    writeln("Walking time to destination: ", EdgeTime[e]);
                }
            }
        }
    }
    writeln();

    var Edg = 0;
    for (var e2 in E)
        Edg = Edg + 1;

    writeln();
    writeln("------------------ Inputs ---------------------------------------");
    writeln();
    writeln("Total number of nodes (N): ", Nodes);
    writeln("Total number of edges (E): ", Edg);
    writeln("Total number of parking lots (L): ", Q);
    writeln("Maximum walking distance (W_km): ", W_km);
    writeln("Maximum parking cost (Cmax): ", Cmax);
    writeln("Source node (S) and destination (D): ", S, " -> ", D);

    writeln();
    writeln("------------------ Results --------------------------------------");
    writeln();
    writeln("Cost of parking: ", Cost);
    writeln("Travel time: ", T[D]);
    writeln("Distance Taken: ", DistanceTaken);
}
