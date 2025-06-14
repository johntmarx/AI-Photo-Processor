import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'

interface ProcessingChartProps {
  data: Array<{
    time: string
    processed: number
    failed: number
  }>
  title?: string
}

export default function ProcessingChart({ data, title = "Processing Activity" }: ProcessingChartProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="time" 
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '6px',
              }}
            />
            <Line
              type="monotone"
              dataKey="processed"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              dot={false}
              name="Processed"
            />
            <Line
              type="monotone"
              dataKey="failed"
              stroke="hsl(var(--destructive))"
              strokeWidth={2}
              dot={false}
              name="Failed"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}