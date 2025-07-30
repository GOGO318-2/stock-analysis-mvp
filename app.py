import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, BarChart3, Star, Newspaper, Info, Target, Lightbulb, X } from 'lucide-react';

const StockAnalyzer = () => {
  const [activeTab, setActiveTab] = useState('search');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentStock, setCurrentStock] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [favorites, setFavorites] = useState([]);
  const [loading, setLoading] = useState(false);

  // 模拟股票数据
  const mockStockData = {
    'TSLA': {
      symbol: 'TSLA',
      name: 'Tesla, Inc.',
      price: 366.26,
      change: 9.74,
      marketCap: 100.0,
      volume: '25.2M',
      pe: 45.6,
      recommendation: 347.95
    },
    'AAPL': {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 189.25,
      change: 2.31,
      marketCap: 2800.0,
      volume: '45.8M',
      pe: 28.4,
      recommendation: 185.50
    },
    'GOOGL': {
      symbol: 'GOOGL',
      name: 'Alphabet Inc.',
      price: 2734.87,
      change: -1.24,
      marketCap: 1750.0,
      volume: '12.3M',
      pe: 24.2,
      recommendation: 2700.00
    }
  };

  // 公开市场数据
  const mockMarketData = [
    {
      symbol: 'CDNS',
      name: 'Cadence Design Systems',
      price: 366.26,
      change: 9.74,
      rating: '低',
      marketCap: 100.0,
      recommendation: 347.95
    },
    {
      symbol: 'SNPS',
      name: 'Synopsys, Inc.',
      price: 635.81,
      change: 7.29,
      rating: '低',
      marketCap: 117.65,
      recommendation: 604.02
    }
  ];

  // 搜索股票
  const handleSearch = (query = searchQuery) => {
    setLoading(true);
    const searchTerm = query || currentStock;
    
    setTimeout(() => {
      const result = mockStockData[searchTerm.toUpperCase()];
      if (result) {
        setSearchResult(result);
        setActiveTab('advice');
      } else {
        setSearchResult(null);
      }
      setLoading(false);
    }, 1000);
  };

  // 生成投资建议
  const generateInvestmentAdvice = (stock) => {
    const isPositive = stock.change > 0;
    return {
      investmentType: isPositive ? '短期持有' : '长期价值',
      targetPrice: (stock.price * 1.15).toFixed(2),
      buyPrice: (stock.price * 0.95).toFixed(2),
      entryPrice: (stock.price * 0.98).toFixed(2),
      positionSize: isPositive ? '适中仓位' : '分批建仓',
      sentiment: isPositive ? '乐观' : '谨慎乐观',
      marketCondition: '当前大部分股票买入等级较低，建议谨慎操作',
      newsSentiment: '新闻情绪中性，关注基本面变化'
    };
  };

  // 添加到收藏
  const addToFavorites = (stock) => {
    if (!favorites.some(f => f.symbol === stock.symbol)) {
      setFavorites([...favorites, stock]);
    }
  };

  // 从收藏中移除
  const removeFavorite = (symbol) => {
    setFavorites(favorites.filter(f => f.symbol !== symbol));
  };

  // 当currentStock变化时自动搜索
  useEffect(() => {
    if (currentStock && activeTab === 'search') {
      handleSearch(currentStock);
    }
  }, [currentStock, activeTab]);

  // 搜索页面
  const SearchPage = () => (
    <div className="p-4">
      <h2 className="text-xl font-bold text-white mb-4">股票分析器</h2>
      <div className="mb-4">
        <div className="text-sm text-gray-400 mb-2">支持港股: 输入XH0700</div>
        <div className="flex space-x-2">
          <input
            type="text"
            value={currentStock || searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentStock(e.target.value);
            }}
            placeholder="输入股票代码"
            className="flex-1 px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
          />
          <button
            onClick={() => handleSearch()}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 text-white px-4 py-2 rounded flex items-center"
          >
            {loading ? (
              <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
            ) : (
              <Search className="w-4 h-4" />
            )}
          </button>
          <button className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded">
            <BarChart3 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {searchResult && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3">
            {searchResult.name} ({searchResult.symbol})
          </h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="text-gray-300">
              价格: <span className="text-white font-medium">${searchResult.price}</span>
            </div>
            <div className="text-gray-300">
              涨幅: <span className={`font-medium ${searchResult.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {searchResult.change}%
              </span>
            </div>
            <div className="text-gray-300">
              市值: <span className="text-white font-medium">{searchResult.marketCap}B</span>
            </div>
            <div className="text-gray-300">
              成交量: <span className="text-white font-medium">{searchResult.volume}</span>
            </div>
          </div>
          <div className="mt-4 flex space-x-2">
            <button
              onClick={() => setActiveTab('advice')}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded text-sm"
            >
              获取投资建议
            </button>
            <button
              onClick={() => addToFavorites(searchResult)}
              className="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded text-sm flex items-center"
            >
              <Star className="w-4 h-4 mr-1" />
              收藏
            </button>
          </div>
        </div>
      )}
    </div>
  );

  // 收藏列表组件
  const FavoritesList = () => (
    <div className="p-4">
      <h3 className="text-lg font-semibold mb-3 text-white flex items-center">
        <Star className="w-5 h-5 mr-2 text-yellow-400" />
        我的收藏
      </h3>
      {favorites.length === 0 ? (
        <p className="text-gray-400 text-sm">暂无收藏股票</p>
      ) : (
        <div className="space-y-2">
          {favorites.map((stock) => (
            <div key={stock.symbol} className="bg-gray-700 p-2 rounded flex justify-between items-center">
              <div className="flex-1">
                <div className="text-white text-sm font-medium">{stock.symbol}</div>
                <div className="text-gray-300 text-xs truncate">{stock.name}</div>
              </div>
              <button
                onClick={() => removeFavorite(stock.symbol)}
                className="text-red-400 hover:text-red-300 ml-2"
                title="移除"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  // 投资建议页面
  const InvestmentAdvice = () => {
    if (!searchResult) return null;

    const advice = generateInvestmentAdvice(searchResult);
    
    return (
      <div className="p-4 max-w-4xl mx-auto">
        <h2 className="text-xl font-bold text-white mb-4">
          {searchResult.name} ({searchResult.symbol}) 投资建议
        </h2>
        
        <div className="bg-gray-800 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
            <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
            投资策略建议
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div className="bg-gray-700 p-3 rounded">
              <div className="text-sm text-gray-400">投资类型</div>
              <div className="text-white font-medium">{advice.investmentType}</div>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <div className="text-sm text-gray-400">目标价位</div>
              <div className="text-white font-medium">${advice.targetPrice}</div>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <div className="text-sm text-gray-400">建议买入价</div>
              <div className="text-white font-medium">${advice.buyPrice}</div>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <div className="text-sm text-gray-400">市场情绪</div>
              <div className="text-white font-medium">{advice.sentiment}</div>
            </div>
          </div>

          <div className="bg-gray-700 p-3 rounded mb-4">
            <div className="text-sm text-gray-400 mb-2">市场情况</div>
            <div className="text-white text-sm">{advice.marketCondition}</div>
          </div>

          <div className="bg-gray-700 p-3 rounded">
            <div className="text-sm text-gray-400 mb-2">新闻情绪</div>
            <div className="text-white text-sm">{advice.newsSentiment}</div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-3 flex items-center">
            <Newspaper className="w-5 h-5 mr-2 text-green-400" />
            相关新闻分析
          </h3>
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="bg-gray-700 p-3 rounded flex justify-between items-center">
                <div className="flex-1">
                  <div className="text-white text-sm">- 中性</div>
                </div>
                <div className="text-gray-400 text-xs">1970-01-01 00:00:00</div>
              </div>
            ))}
          </div>
          
          <button className="mt-4 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded text-sm flex items-center">
            <Star className="w-4 h-4 mr-2" />
            收藏
          </button>
        </div>
      </div>
    );
  };

  // 公开市场页面
  const PublicMarket = () => (
    <div className="p-4 max-w-6xl mx-auto">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-white mb-2">公开市场</h2>
        <div className="bg-blue-900 p-3 rounded text-sm text-blue-100">
          <Info className="w-4 h-4 inline mr-2" />
          市场提醒: 当前大部分股票买入等级较低，建议谨慎操作，可考虑关注基本面较好的个股进行长期布局。
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {mockMarketData.map((stock) => (
          <div key={stock.symbol} className="bg-gray-800 rounded-lg p-4">
            <div className="flex justify-between items-start mb-3">
              <div className="flex-1">
                <div className="flex items-center mb-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                  <h3 className="font-semibold text-white text-sm">{stock.symbol} - {stock.name}</h3>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="text-gray-300">
                    <span className="text-yellow-400">💰</span> 价格: ${stock.price}
                  </div>
                  <div className="text-gray-300">
                    <span className="text-blue-400">📊</span> 涨幅: {stock.change}%
                  </div>
                  <div className="text-gray-300">
                    <span className="text-green-400">⭐</span> 买入等级: {stock.rating}
                  </div>
                  <div className="text-gray-300">
                    <span className="text-purple-400">💼</span> 市值: {stock.marketCap}B
                  </div>
                </div>
                {stock.recommendation && (
                  <div className="mt-2 text-xs">
                    <span className="text-red-400">📈</span> 建议买入价: ${stock.recommendation}
                  </div>
                )}
              </div>
              <div className="flex flex-col space-y-2 ml-2">
                <button
                  onClick={() => addToFavorites(stock)}
                  disabled={favorites.some(f => f.symbol === stock.symbol)}
                  className={`px-2 py-1 rounded text-xs ${
                    favorites.some(f => f.symbol === stock.symbol)
                      ? 'bg-yellow-600 text-white cursor-not-allowed'
                      : 'bg-gray-700 hover:bg-yellow-600 text-white'
                  }`}
                >
                  <Star className="w-3 h-3" />
                </button>
                <button
                  onClick={() => {
                    setCurrentStock(stock.symbol);
                    setActiveTab('search');
                  }}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-2 py-1 rounded text-xs"
                >
                  <Search className="w-3 h-3" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-6 text-center">
        <div className="bg-gray-800 p-3 rounded text-sm text-gray-300">
          应用运行出错: There are multiple elements with the same key='view_SNPS_1'. To fix this, please make sure that the key argument is unique for each element you create.
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 flex">
      {/* 左侧边栏 */}
      <div className="w-80 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-4 border-b border-gray-700">
          <h1 className="text-xl font-bold text-white flex items-center">
            <BarChart3 className="w-6 h-6 mr-2 text-blue-400" />
            股票分析器
          </h1>
        </div>

        {/* 搜索区域 */}
        <div className="p-4 border-b border-gray-700">
          <div className="text-sm text-gray-400 mb-2">支持港股: 输入XH0700</div>
          <div className="flex space-x-2">
            <input
              type="text"
              value={currentStock || searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentStock(e.target.value);
              }}
              placeholder="输入股票代码"
              className="flex-1 px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-blue-500 focus:outline-none text-sm"
            />
            <button
              onClick={() => handleSearch()}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 text-white px-3 py-2 rounded"
            >
              {loading ? (
                <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
              ) : (
                <Search className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>

        {/* 收藏列表 */}
        <div className="flex-1 overflow-y-auto">
          <FavoritesList />
        </div>

        {/* 页面导航 */}
        <div className="p-4 border-t border-gray-700">
          <h3 className="text-sm font-semibold text-white mb-3">页面导航</h3>
          <div className="space-y-2">
            <button
              onClick={() => setActiveTab('search')}
              className={`w-full text-left px-3 py-2 rounded text-sm flex items-center ${
                activeTab === 'search' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              <Search className="w-4 h-4 mr-2" />
              首页
            </button>
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`w-full text-left px-3 py-2 rounded text-sm flex items-center ${
                activeTab === 'dashboard' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              <BarChart3 className="w-4 h-4 mr-2" />
              基本面
            </button>
            <button
              onClick={() => setActiveTab('advice')}
              className={`w-full text-left px-3 py-2 rounded text-sm flex items-center ${
                activeTab === 'advice' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              <Target className="w-4 h-4 mr-2" />
              投资建议
            </button>
            <button
              onClick={() => setActiveTab('market')}
              className={`w-full text-left px-3 py-2 rounded text-sm flex items-center ${
                activeTab === 'market' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'
              }`}
            >
              <TrendingUp className="w-4 h-4 mr-2" />
              公共市场
            </button>
          </div>
        </div>
      </div>

      {/* 主内容区域 */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'search' && <SearchPage />}
        {activeTab === 'advice' && <InvestmentAdvice />}
        {activeTab === 'market' && <PublicMarket />}
        {activeTab === 'dashboard' && (
          <div className="p-4">
            <h2 className="text-xl font-bold text-white mb-4">基本面分析</h2>
            <div className="bg-gray-800 rounded-lg p-4">
              <p className="text-gray-300">基本面分析功能开发中...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockAnalyzer;
